"""Calculates exact match (EM) and F1 for benchmarking QA models.

Code is borrowed from "evaluation_eval.py" in the MRQA 2019 Shared Task
repository (https://github.com/mrqa/MRQA-Shared-Task-2019).

Usage:

    python3 evaluate.py \
        --dataset_path "datasets/squad_dev.jsonl.gz" \
        --output_path "squad_predictions.txt"

Author:
    Shrey Desai
"""

import argparse
import gzip
import json
import re
import string
from collections import Counter


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.

    Args:
        s: String to normalize.

    Returns:
        Cleaned string with lowercase, no punctuations, no articles, and
            and extraneous whitespace.
    """

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """Calculates F1 score.

    Args:
        prediction: Predicted answer span (string).
        ground_truth: True answer span (string).

    Returns:
        F1 score.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """Calculates exact match (EM) score.

    Args:
        prediction: Predicted answer span (string).
        ground_truth: True answer span (string).

    Returns:
        EM score.
    """
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    Maximizes metric function over available gold spans. Because there can be
    multiple legal answers, we do not penalize the model by only testing on
    the first gold span.

    Args:
        metric_fn: Function to maximize over.
        prediction: Predicted answer span (string).
        ground_truths: List of true answer spans (each string).

    Returns:
        Max score.
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def read_predictions(prediction_file):
    """Reads line-delimited predictions from output file.

    Args:
        prediction_file: Path to output file (string).

    Returns:
        Predictions dict mapping question id (qid) to answer span.
    """
    predictions = {}
    with open(prediction_file) as f:
        for line in f:
            example = json.loads(line)
            predictions[example['qid']] = example['answer']
    return predictions


def read_answers(gold_file):
    """Reads answers from dataset file. Each question (marked by its qid)
    can have multiple possible answer spans.

    Args:
        gold_file: Path to dataset file (string).

    Returns:
        True dict mapping question id (id) to answer span(s).
    """
    answers = {}
    with gzip.open(gold_file, 'rb') as f:
        for i, line in enumerate(f):
            example = json.loads(line)
            if i == 0 and 'header' in example:
                continue
            for qa in example['qas']:
                answers[qa['qid']] = qa['answers']
    return answers


def evaluate(answers, predictions, skip_no_answer=False):
    """Main function for evaluating predicted answers.

    Args:
        answers: Dict of qid -> gold answer span(s)
        predictions: Dict of qid -> predicted answer span
        skip_no_answer: Whether to skip unanswered questions or not. By default,
            this is disabled, so unanswered questions will receive 0 EM/F1.

    Returns:
        EM and F1 maximized over gold answer spans.
    """
    f1 = exact_match = total = 0
    for qid, ground_truths in answers.items():
        if qid not in predictions:
            if not skip_no_answer:
                message = 'Unanswered question %s will receive score 0.' % qid
                print(message)
                total += 1
            continue
        total += 1
        prediction = predictions[qid]
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)

    exact_match = round(100.0 * exact_match / total, 2)
    f1 = round(100.0 * f1 / total, 2)

    return {'EM': exact_match, 'F1': f1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help='path to evaluation dataset')
    parser.add_argument('--output_path', type=str, help='path to output predictions')
    parser.add_argument('--skip_no_answer', action='store_true')
    args = parser.parse_args()

    answers = read_answers(args.dataset_path)
    predictions = read_predictions(args.output_path)
    metrics = evaluate(answers, predictions, args.skip_no_answer)
    print(metrics)
