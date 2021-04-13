import copy
import math
from collections import defaultdict
import pandas as pd
import numpy as np


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        # raise ValueError('Relevance score length < k')
        return np.sum(r) / k
    return np.mean(r)

def average_precision(r):
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)

def mean_average_precision(rs):
    return np.mean([average_precision(r) for r in rs])

def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


###############################################

def evaluate(df):
    def match_section(qsecs, match_sec):
        for qsec in qsecs:
            qsec = qsec.split('_')[:3]
            msec = match_sec.split('_')[:3]
            ll = len(qsec) if len(qsec) < len(msec) else len(msec)
            if qsec[:ll] == msec[:ll]:
                return 1
        return 0

    def match_chapter(qsecs, match_sec):
        for qsec in qsecs:
            qsec = qsec.split('_')[:2]
            msec = match_sec.split('_')[:2]
            ll = len(qsec) if len(qsec) < len(msec) else len(msec)
            if qsec[:ll] == msec[:ll]:
                return 1
        return 0

    match_methods = [i for i in df.columns if i.endswith('_match')]

    read_until_now = set()
    v_section = defaultdict(list)
    v_chapter = defaultdict(list)
    # v_his_section = defaultdict(list)
    for _, row in df.fillna('').iterrows():
        q = row['quiz_id']
        qsec = row['quiz_section']
        # qsec = row['quiz_sec']
        outcome = row['outcome']

        if not q.startswith('q') or outcome != '0':# not in ('0', 0):
            continue

        for method in match_methods:
            m = row[method]
            if not m:
                m = '[]'
            m = eval(m)
            v_section[method].append([match_section([qsec], str(i)) for i in m])
            v_chapter[method].append([match_chapter([qsec], str(i)) for i in m])
            # v_his_section[method].append([match_section(read_until_now, i) for i in m])

    def eval_mean(func_eval):
        def f(row):
            return np.mean([func_eval(l) for l in row])
        return f

    eval_methods = (
        ("MRR", mean_reciprocal_rank),
        ("DCG@1", eval_mean(lambda x: dcg_at_k(x, 1))),
        ("DCG@3", eval_mean(lambda x: dcg_at_k(x, 3))),
        ("DCG@5", eval_mean(lambda x: dcg_at_k(x, 5))),
        # ("MAP@1", eval_mean(lambda x: average_precision(x[:1]))),
        # ("MAP@3", eval_mean(lambda x: average_precision(x[:3]))),
        # ("MAP@5", eval_mean(lambda x: average_precision(x[:5]))),
        # ("Precision@1", eval_mean(lambda x: precision_at_k(x, 1))),
        # ("Precision@3", eval_mean(lambda x: precision_at_k(x, 3))),
        # ("Precision@5", eval_mean(lambda x: precision_at_k(x, 5))),
    )

    d_sec = {}
    d_chapter = {}
    for eval_name, eval_row in eval_methods:
            for method in match_methods:
                r = eval_row(v_section[method])
                d_sec[(eval_name, method.replace('_match', ''))] = r

                r = eval_row(v_chapter[method])
                d_chapter[(eval_name, method.replace('_match', ''))] = r

    return d_sec, d_chapter

def evaluate_file(match_file):
    cum_sec = {}
    cmu_chapter = {}

    df = pd.read_csv(match_file, encoding='utf8')
    d_sec, d_chapter = evaluate(df)

    for key in d_sec:
        if key not in cum_sec:
            cum_sec[key] = []
        cum_sec[key].append(d_sec[key])

    for key in d_chapter:
        if key not in cmu_chapter:
            cmu_chapter[key] = []
        cmu_chapter[key].append(d_chapter[key])
    print(cum_sec)
    pd.DataFrame(cum_sec).round(4).to_excel("evaluate.xlsx")


