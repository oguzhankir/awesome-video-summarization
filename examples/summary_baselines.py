#!/usr/bin/env python3
"""Educational summary decoding on an explicit frame timeline; no model/data downloads.

Intervals are half-open [start, end). Knapsack ties keep earlier solutions.
This is not the canonical evaluator for any published benchmark.
"""
import argparse
import json
import math
import random


def knapsack(lengths, values, capacity):
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0:
        raise ValueError('capacity must be a nonnegative integer')
    if len(lengths) != len(values):
        raise ValueError('lengths and values must have equal sizes')
    if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in lengths):
        raise ValueError('lengths must be positive integers')
    if any(not math.isfinite(v) for v in values):
        raise ValueError('values must be finite')
    rows = [[0.0] * (capacity + 1)]
    for length, value in zip(lengths, values):
        previous = rows[-1]
        current = previous.copy()
        for budget in range(length, capacity + 1):
            current[budget] = max(previous[budget], previous[budget-length] + value)
        rows.append(current)
    selected = []
    remaining = capacity
    for i in range(len(lengths), 0, -1):
        if rows[i][remaining] > rows[i-1][remaining]:
            selected.append(i-1)
            remaining -= lengths[i-1]
    return list(reversed(selected))


def decode(scores, segments, budget_fraction=0.15, pooling='mean'):
    if not 0 <= budget_fraction <= 1 or not math.isfinite(budget_fraction):
        raise ValueError('budget_fraction must be in [0, 1]')
    if pooling not in {'mean', 'sum'}:
        raise ValueError('pooling must be mean or sum')
    if any(not math.isfinite(s) for s in scores):
        raise ValueError('scores must be finite')
    previous = 0
    lengths, values = [], []
    for start, end in segments:
        if any(not isinstance(i, int) or isinstance(i, bool) for i in (start,end)) or start != previous or not start < end <= len(scores):
            raise ValueError('segments must partition the timeline using half-open intervals')
        lengths.append(end-start)
        values.append(sum(scores[start:end]) / (end-start if pooling == 'mean' else 1))
        previous = end
    if previous != len(scores):
        raise ValueError('segments must cover the complete score timeline')
    chosen = knapsack(lengths, values, math.floor(budget_fraction * len(scores)))
    mask = [0] * len(scores)
    for i in chosen:
        start,end = segments[i]
        mask[start:end] = [1] * (end-start)
    return mask, chosen


def temporal_f1(prediction, references, aggregation='mean'):
    if aggregation not in {'mean','max'} or not references:
        raise ValueError('provide references and aggregation mean or max')
    if any(v not in (0,1) for v in prediction):
        raise ValueError('prediction must be binary')
    values = []
    for reference in references:
        if len(reference) != len(prediction) or any(v not in (0,1) for v in reference):
            raise ValueError('references must be binary and aligned')
        denominator = sum(prediction) + sum(reference)
        values.append(2 * sum(a*b for a,b in zip(prediction,reference)) / denominator if denominator else 0.0)
    return sum(values)/len(values) if aggregation == 'mean' else max(values)


def uniform_keyframes(frame_count, count):
    if not isinstance(frame_count, int) or not isinstance(count, int) or not 0 <= count <= frame_count:
        raise ValueError('require integer 0 <= count <= frame_count')
    return [((2*i+1)*frame_count)//(2*count) for i in range(count)]


def demo():
    rng = random.Random(17)
    scores = [rng.random() for _ in range(40)]
    segments = [(0,3),(3,6),(6,10),(10,15),(15,25),(25,40)]
    references = [[int(i < 6) for i in range(40)], [int(10 <= i < 15) for i in range(40)]]
    results = {}
    for name,sequence in [('constant',[1.0]*40),('random',scores)]:
        for pooling in ('mean','sum'):
            mask, selected = decode(sequence,segments,pooling=pooling)
            results[f'{name}_{pooling}'] = {'shots':selected,'frames':sum(mask),'f1_mean':round(temporal_f1(mask,references),4),'f1_max':round(temporal_f1(mask,references,'max'),4)}
    return {'note':'Synthetic teaching example; these are not SumMe or TVSum results.',
            'budget_frames':6,'uniform_storyboard_indices':uniform_keyframes(40,6),
            'greedy_counterexample_optimal_shots':knapsack([6,5,5],[9,8,8],10),
            'results':results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true', help='run deterministic synthetic demonstration')
    args = parser.parse_args()
    if not args.demo:
        parser.error('use --demo, or import these functions with your own aligned data')
    print(json.dumps(demo(),indent=2))
