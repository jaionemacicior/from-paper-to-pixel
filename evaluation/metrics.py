import editdistance
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# CER
def cer(pred: str, gt: str) -> float:
    if len(gt) == 0:
        return float('inf')
    return editdistance.eval(pred, gt) / len(gt)

# WER
def wer(pred: str, gt: str) -> float:
    pred_words = pred.split()
    gt_words = gt.split()
    if len(gt_words) == 0:
        return float('inf')
    return editdistance.eval(pred_words, gt_words) / len(gt_words)

# Levenshtein Distance (LEV)
def lev(pred: str, gt: str) -> int:
    return editdistance.eval(pred, gt)

# Normalized Edit Distance (NED)
def ned(pred: str, gt: str) -> float:
    max_len = max(len(pred), len(gt))
    if max_len == 0:
        return 0.0
    return lev(pred, gt) / max_len

# BLEU score
def bleu(pred: str, gt: str) -> float:
    reference = [gt.split()]
    hypothesis = pred.split()
    return sentence_bleu(reference, hypothesis, smoothing_function=SmoothingFunction().method1)

# ROUGE scores (ROUGE-1, ROUGE-L)
def rouge(pred: str, gt: str) -> dict:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(gt, pred)
    return {
        "rouge1": scores['rouge1'].fmeasure,
        "rougeL": scores['rougeL'].fmeasure
    }

def iou(box1, box2):
    """IoU entre dos cajas [x1,y1,x2,y2]"""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB-xA) * max(0, yB-yA)
    box1Area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2Area = (box2[2]-box2[0])*(box2[3]-box2[1])
    return interArea / float(box1Area + box2Area - interArea + 1e-6)