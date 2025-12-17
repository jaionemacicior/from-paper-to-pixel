import os
import gc
import time
import argparse
import json
import torch
from PIL import Image
import pandas as pd
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
import logging
logging.getLogger("codecarbon").disabled = True
from codecarbon import EmissionsTracker

# -----------------------------
# 1. Memory and attention config
# -----------------------------
try:
    import flash_attn
    print("FlashAttention is installed")
    USE_FLASH_ATTENTION = True
except ImportError:
    print("FlashAttention is not installed")
    USE_FLASH_ATTENTION = False

USE_QLORA = True
USE_LORA = True

SYSTEM_PROMPT = """You are an OCR expert specialized in Spanish documents.
You are analyzing an old book scan with potentially low quality.

INSTRUCTIONS:

    Extract ALL text exactly as it appears.

    Do not correct, interpret or modify the text in any way.

    Return ONLY the raw text, without any additional comments or formatting.

    Do not invent content not present in the image.

The output must be EXACTLY the recognized text, without adding anything else.
"""

USER_PROMPT = "Please perform OCR on this Spanish document."

# -----------------------------
# 2. Memory cleanup
# -----------------------------
def clear_memory(safe_clear=True):
    """
    Frees GPU memory and performs garbage collection.
    """
    protected_vars = {'trainer': None, 'model': None, 'processor': None}
    if safe_clear:
        for var in protected_vars:
            if var in globals():
                protected_vars[var] = globals()[var]

    for var in ['inputs', 'peft_model', 'bnb_config', 'batch']:
        if var in globals():
            del globals()[var]

    time.sleep(1)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(1)

    if safe_clear:
        for var, val in protected_vars.items():
            if val is not None:
                globals()[var] = val

    print(f"\nMemory status after cleanup:")
    print(f"- Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"- Reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

# -----------------------------
# 3. Load and prepare dataset
# -----------------------------
def load_and_prepare_data(corpus_name):
    """
    Loads images and OCR text from a corpus folder under 'docLayout/'.
    Args:
        corpus_name (str): Name of the corpus folder.
    Returns:
        dict: {'train': [...], 'val': [...]} chat examples.
    """
    print(f"\nLoading OCR dataset for corpus '{corpus_name}'...")
    train_image_dir = os.path.join('granite', corpus_name, 'train')
    train_ocr_dir = os.path.join('data', corpus_name, 'train', 'txt')

    val_image_dir = os.path.join('granite', corpus_name, 'val')
    val_ocr_dir = os.path.join('data', corpus_name, 'val', 'txt')

    train, val = [], []

    # Load training data
    for fname in sorted(os.listdir(train_image_dir)):
        if not fname.endswith(".jpg"):
            continue
        image_path = os.path.join(train_image_dir, fname)
        text_path = os.path.join(train_ocr_dir, fname.replace(".jpg", ".txt"))
        if not os.path.exists(text_path):
            continue
        with open(text_path, encoding="utf-8") as f:
            ocr_text = f.read().strip()
        image = Image.open(image_path).convert("RGB")
        chat = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": ocr_text}]},
        ]
        train.append(chat)

    print(f"Total training examples loaded: {len(train)}")

    # Load validation data
    for fname in sorted(os.listdir(val_image_dir)):
        if not fname.endswith(".jpg"):
            continue
        image_path = os.path.join(val_image_dir, fname)
        text_path = os.path.join(val_ocr_dir, fname.replace(".jpg", ".txt"))
        if not os.path.exists(text_path):
            continue
        with open(text_path, encoding="utf-8") as f:
            ocr_text = f.read().strip()
        image = Image.open(image_path).convert("RGB")
        chat = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": ocr_text}]},
        ]
        val.append(chat)

    print(f"Total validation examples loaded: {len(val)}")
    return {"train": train, "val": val}

# -----------------------------
# 4. Model setup
# -----------------------------
def setup_model(input_model):
    """
    Loads the model and processor, applies LoRA/QLoRA if enabled.
    """
    print("\nSetting up the model...")

    bnb_config = None
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["vision_tower", "lm_head"],
            llm_int8_enable_fp32_cpu_offload=True
        )

    model = AutoModelForImageTextToText.from_pretrained(
        input_model,
        device_map="auto",
        dtype=torch.bfloat16,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2" if USE_FLASH_ATTENTION else "sdpa",
        low_cpu_mem_usage=True
    )
    model.config.use_cache = False

    peft_config = None
    if USE_LORA:
        peft_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=[name for name, _ in model.named_modules()
                            if 'language_model' in name and '_proj' in name],
            use_dora=True,
            init_lora_weights="gaussian"
        )
        model = get_peft_model(model, peft_config)
        model.enable_input_require_grads()
        model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(input_model)

    print(f"\nModel setup complete: {'with LoRA' if USE_LORA else 'without LoRA'} | "
          f"{'with QLoRA' if USE_QLORA else 'without QLoRA'}")

    return model, processor, peft_config

# -----------------------------
# 5. Training configuration
# -----------------------------
def get_training_config(output_model_dir):
    return SFTConfig(
        output_dir=output_model_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=1,
        optim="adamw_torch_fused",
        bf16=True,
        push_to_hub=False,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

# -----------------------------
# 6. Collate function
# -----------------------------
def collate_fn(examples, processor):
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    image_inputs = [example[1]['content'][0]['image'].convert('RGB') for example in examples]

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    assistant_tokens = processor.tokenizer("<|assistant|>", return_tensors="pt")['input_ids'][0]
    eos_token = processor.tokenizer("<|end_of_text|>", return_tensors="pt")['input_ids'][0]

    for i in range(batch["input_ids"].shape[0]):
        apply_loss = False
        for j in range(batch["input_ids"].shape[1]):
            if not apply_loss:
                labels[i][j] = -100
            if ((j >= len(assistant_tokens) + 1) and
                    torch.all(batch["input_ids"][i][j + 1 - len(assistant_tokens):j + 1] == assistant_tokens)):
                apply_loss = True
            if batch["input_ids"][i][j] == eos_token:
                apply_loss = False

    batch["labels"] = labels
    return batch

# -----------------------------
# 7. Main
# -----------------------------
def finetune_model(input_model, output_model, corpus):
    output_dir = os.path.join('granite', 'models', output_model)
    os.makedirs(output_dir, exist_ok=True)

    data = load_and_prepare_data(corpus)
    model, processor, peft_config = setup_model(input_model)
    clear_memory()
    training_config = get_training_config(output_model)

    carbon_config = {
        "save_to_file": True,
        "log_level": "ERROR",
        "tracking_mode": "process",
        "output_dir": output_dir,
        "measure_power_secs": 1,
        "save_to_api": False,
        "allow_multiple_runs": True
    }
    tracker = EmissionsTracker(**carbon_config)
    tracker.start()

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=data['train'],
        eval_dataset=data['val'],
        data_collator=lambda x: collate_fn(x, processor),
        peft_config=peft_config,
    )

    print("\nStarting training...")
    try:
        trainer.train()
        print("\nTraining completed successfully!")
        trainer.save_model(training_config.output_dir)

        if peft_config:
            model = model.merge_and_unload()
            model.save_pretrained(training_config.output_dir)

    except Exception as e:
        print(f"Error during training: {e}")

    finally:
        tracker.stop()
        emissions_file = os.path.join(output_model, "emissions.json")
        if os.path.exists(emissions_file):
            df = pd.read_csv(emissions_file)
            last_row = df.iloc[-1]
            emissions_data = {col: str(last_row[col]) for col in [
                "duration", "emissions", "cpu_energy", "gpu_energy", "ram_energy",
                "energy_consumed", "cpu_count", "gpu_count", "cpu_model", "gpu_model",
                "ram_total_size", "country_iso_code"
            ]}
            with open(emissions_file, "w") as f:
                json.dump(emissions_data, f, indent=4)
        clear_memory(safe_clear=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True,
                        help="Name of the corpus folder inside 'docLayout/'")
    parser.add_argument("--input_model", type=str, default='ibm-granite/granite-vision-3.2-2b',
                        help="Pretrained model to fine-tune")
    parser.add_argument("--output_model", type=str, required=True,
                        help="Directory to save fine-tuned model")
    args = parser.parse_args()
    finetune_model(input_model=args.input_model, output_model=args.output_model, corpus=args.corpus)
