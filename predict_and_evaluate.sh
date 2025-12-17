#!/bin/bash
# eval_all.sh - Interactive evaluation launcher (dynamic models)

# ==============================================
# COLORS
# ==============================================
GREEN="\e[32m"
YELLOW="\e[33m"
CYAN="\e[36m"
RED="\e[31m"
RESET="\e[0m"

# ==============================================
# FUNCTION TO DETECT AVAILABLE MODELS
# ==============================================
# Reads subdirectories inside:  <model_type>/models/
get_models() {
    local model_type="$1"
    local path="$model_type/models"

    if [ ! -d "$path" ]; then
        echo ""
        return
    fi

    # Always include baseline
    local models=("baseline")

    # Add all folders that exist inside models/
    while IFS= read -r d; do
        models+=("$d")
    done < <(find "$path" -maxdepth 1 -mindepth 1 -type d -printf "%f\n")

    printf "%s\n" "${models[@]}"
}

get_tesseract_models() {
    local path="tesseract/models"
    local models=()

    if [ ! -d "$path" ]; then
        echo ""
        return
    fi

    # Add folders EXCEPT configs
    while IFS= read -r d; do
        models+=("$d")
    done < <(find "$path" -maxdepth 1 -mindepth 1 -type d -printf "%f\n" \
             | grep -v "^configs$")

    # Add .traineddata files without extension
    while IFS= read -r f; do
        models+=("$f")
    done < <(find "$path" -maxdepth 1 -type f -name "*.traineddata" -printf "%f\n" \
             | sed 's/.traineddata$//')

    printf "%s\n" "${models[@]}"
}

# Load lists dynamically based on your structure
GRANITE_MODELS=($(get_models "granite"))
TROCR_MODELS=($(get_models "trOCR"))
TESSERACT_MODELS=($(get_tesseract_models))
DOCLAYOUT_MODELS=($(get_models "docLayout"))

SPLITS=("val" "test" "train")


# ==============================================
# ASK CORPUS
# ==============================================
echo -e "${CYAN}Enter the corpus name to evaluate:${RESET}"
read -r CORPUS

if [ -z "$CORPUS" ]; then
    echo -e "${RED}Error: No corpus provided. Exiting.${RESET}"
    exit 1
fi

echo -e "${GREEN}Corpus selected:${RESET} $CORPUS"
echo ""


# ==============================================
# SHOW AVAILABLE MODEL FAMILIES
# ==============================================
echo -e "${CYAN}Detected model families and models:${RESET}"

[[ ${#GRANITE_MODELS[@]} -gt 0 ]] && echo "  1) Granite:    ${GRANITE_MODELS[*]}"
[[ ${#TROCR_MODELS[@]} -gt 0 ]] && echo "  2) TrOCR:      ${TROCR_MODELS[*]}"
[[ ${#TESSERACT_MODELS[@]} -gt 0 ]] && echo "  3) Tesseract:  ${TESSERACT_MODELS[*]}"
[[ ${#DOCLAYOUT_MODELS[@]} -gt 0 ]] && echo "  4) DocLayout:  ${DOCLAYOUT_MODELS[*]}"

echo "  5) All families"
echo ""
echo "Enter numbers separated by spaces (e.g., 1 3 4):"
read -r choices
echo ""

SELECTED_MODELS=($choices)

should_run() {
    local target=$1
    for choice in "${SELECTED_MODELS[@]}"; do
        if [[ "$choice" == "$target" || "$choice" == "5" ]]; then
            return 0
        fi
    done
    return 1
}


# ==============================================
# FUNCTION TO RUN EVAL
# ==============================================
run_eval() {
    local model_type=$1
    local model_name=$2
    local split=$3

    echo -e "${YELLOW}"
    echo "=========================================="
    echo " Evaluating: $model_type | $model_name | $split"
    echo " Corpus: $CORPUS"
    echo "=========================================="
    echo -e "${RESET}"

    python -m evaluation.run_evaluation \
        --model_type "$model_type" \
        --model_name "$model_name" \
        --split "$split" \
        --eval True \
        --corpus "$CORPUS"
}


# ==============================================
# EXECUTION BY FAMILY
# ==============================================

if should_run "1" && [[ ${#GRANITE_MODELS[@]} -gt 0 ]]; then
    echo -e "${GREEN}Running Granite models...${RESET}"
    for m in "${GRANITE_MODELS[@]}"; do
        for s in "${SPLITS[@]}"; do
            run_eval "granite" "$m" "$s"
        done
    done
fi

if should_run "2" && [[ ${#TROCR_MODELS[@]} -gt 0 ]]; then
    echo -e "${GREEN}Running TrOCR models...${RESET}"
    for m in "${TROCR_MODELS[@]}"; do
        for s in "${SPLITS[@]}"; do
            run_eval "trOCR" "$m" "$s"
        done
    done
fi

if should_run "3" && [[ ${#TESSERACT_MODELS[@]} -gt 0 ]]; then
    echo -e "${GREEN}Running Tesseract models...${RESET}"
    for m in "${TESSERACT_MODELS[@]}"; do
        for s in "${SPLITS[@]}"; do
            run_eval "tesseract" "$m" "$s"
        done
    done
fi

if should_run "4" && [[ ${#DOCLAYOUT_MODELS[@]} -gt 0 ]]; then
    echo -e "${GREEN}Running DocLayout models...${RESET}"
    for m in "${DOCLAYOUT_MODELS[@]}"; do
        for s in "${SPLITS[@]}"; do
            run_eval "docLayout" "$m" "$s"
        done
    done
fi

echo ""
echo -e "${GREEN}All evaluations completed ${RESET}"

