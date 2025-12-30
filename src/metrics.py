import string
import re
import numpy as np

from typing import List, Dict


def _normalize_answer(text, punc_chars=string.punctuation, punc_repl=""):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(s):
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def replace_punctuation(s):
        to_replace = set(punc_chars)
        return "".join(punc_repl if ch in to_replace else ch for ch in s)

    def white_space_fix(s):
        return " ".join(s.split())

    text = text.lower()
    text = replace_punctuation(text)
    text = remove_articles(text)
    text = white_space_fix(text)

    return text


def normalize_squad(answer):
    """Normalization used in official SQuAD evaluation script."""
    return _normalize_answer(answer, punc_chars=string.punctuation, punc_repl="")


def get_f1_score(correct, final_answers, final_responses, label):

    recall, precision = correct / (final_answers.count(label) + 1e-20), correct / (final_responses.count(label) + 1e-20) 
    f1 = 2 * (recall * precision) / (recall + precision)

    return f1


# def _f1_score(target, prediction):
#     """Computes token f1 score for a single target and prediction."""
#     prediction_tokens = prediction.split()
#     target_tokens = target.split()
#     common = (collections.Counter(prediction_tokens) &
#               collections.Counter(target_tokens))
#     num_same = sum(common.values())
#     if num_same == 0:
#         return 0
#     precision = 1.0 * num_same / len(prediction_tokens)
#     recall = 1.0 * num_same / len(target_tokens)
#     f1 = (2 * precision * recall) / (precision + recall)
#     return f1


def compute_acc_and_f1(labels: List[str], golds: List[str], preds: List[str]) -> Dict[str, float]:
    """Computes SQuAD metrics, maximizing over answers per question.
    Args:
    golds: list of lists of strings
    preds: list of strings

    Returns:
    dict with score_key: squad score across all golds and predictions
    """

    golds = [normalize_squad(gold) for gold in golds]
    preds = [normalize_squad(pred) for pred in preds]

    cnt_label_correct = {label:0 for label in labels}
    correct = []
    for idx, (gold, pred) in enumerate(zip(golds, preds)):
        if gold == pred:
            cnt_label_correct[pred] += 1
            correct.append(1)
        else:
            correct.append(0)
        # if pred.startswith(gold):
        #     cnt_label_correct[gold] += 1
        #     correct.append(1)
        # else:
        #     correct.append(0)
    accuracy = correct.count(1) / len(correct)

    f1_scores = []
    for label in labels:
        f1 = get_f1_score(cnt_label_correct[label], golds, preds, label)
        f1_scores.append(f1)

    f1_score = sum(f1_scores) / len(f1_scores)

    return {'acc': round(accuracy * 100, 2), 'f1': round(f1_score * 100, 2)}