from __future__ import annotations

from typing import Dict, List

import torch


def accuracy_from_counts(correct: int, total: int) -> float:
    return correct / max(total, 1)


def macro_f1_from_predictions(y_true: List[int], y_pred: List[int], num_classes: int) -> float:
    f1_scores = []
    for cls in range(num_classes):
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp == cls)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != cls and yp == cls)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp != cls)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    return sum(f1_scores) / max(len(f1_scores), 1)


def confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: int) -> List[List[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for yt, yp in zip(y_true, y_pred):
        matrix[yt][yp] += 1
    return matrix
