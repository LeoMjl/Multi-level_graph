from __future__ import annotations

import re
import string
from collections import Counter


def normalize(text: object) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, answers: list[str]) -> float:
    pred = normalize(prediction)
    return 1.0 if answers and any(pred == normalize(answer) for answer in answers) else 0.0


def f1(prediction: str, answers: list[str]) -> float:
    pred_tokens = normalize(prediction).split()
    if not pred_tokens or not answers:
        return 0.0
    pred_counter = Counter(pred_tokens)
    best = 0.0
    for answer in answers:
        gold_tokens = normalize(answer).split()
        if not gold_tokens:
            continue
        common = pred_counter & Counter(gold_tokens)
        overlap = sum(common.values())
        if not overlap:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def rouge_l(prediction: str, answers: list[str]) -> float:
    pred = normalize(prediction).split()
    if not pred or not answers:
        return 0.0
    best = 0.0
    for answer in answers:
        gold = normalize(answer).split()
        if not gold:
            continue
        lcs = longest_common_subsequence(pred, gold)
        precision = lcs / len(pred)
        recall = lcs / len(gold)
        if precision + recall:
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def longest_common_subsequence(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, start=1):
            current.append(prev[j - 1] + 1 if token_a == token_b else max(prev[j], current[-1]))
        prev = current
    return prev[-1]
