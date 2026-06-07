from collections import defaultdict
from typing import List, Dict, Tuple
import os
import sys
import json
import numpy as np
# ---- pycocoevalcap scorers ----
# (You already have these since compute_caption_score runs.)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cococaption')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pycocoevalcap')))

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.spice.spice import Spice  # optional; requires Java

class CocoPairScorer:
    """
    Score ONE (reference, prediction) pair with a chosen COCO metric.
    Reuses scorers efficiently across many pairs.
    """
    def __init__(self, metric: str):
        m = metric.lower()
        self.metric = m
        if m in ("bleu", "bleu1", "bleu4"):
            self.scorer = Bleu(4)      # returns 4 BLEU scores (1..4-gram)
        elif m == "meteor":
            self.scorer = Meteor()
        elif m in ("rougel", "rouge_l"):
            self.scorer = Rouge()
        elif m == "cider":
            self.scorer = Cider()
        # elif m == "spice":
        #     self.scorer = Spice()
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def score(self, ref: str, pred: str) -> float:
        # pycocoevalcap expects dict[int] -> [str]
        gts = {0: [ref or ""]}     # list of refs allowed; we pass single
        res = {0: [pred or ""]}    # single hypothesis
        score, _ = self.scorer.compute_score(gts, res)
        if self.metric in ("bleu", "bleu1", "bleu4"):
            # Bleu.compute_score returns a list of 4 BLEU scores
            if self.metric in ("bleu", "bleu4"):
                return float(score[3])   # BLEU-4
            else:
                return float(score[0])   # BLEU-1
        # Meteor/Rouge/Cider return a single float
        return float(score)


def topk_classification_accuracy_coco(
    records: List[Dict],
    k: int = 3,
    metric: str = "meteor",
    require_min_group: int = 2,
    strict_align: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """
    records: [{"id": <str>, "prediction": <str>, "reference": <str>}, ...]
    For each id with >= require_min_group records, treat refs as g1..gN and preds as p1..pN.
    For each gi, compute scores to all pj; if the "true" pj (j=i by default) is in top-k, count success.
    """
    scorer = CocoPairScorer(metric)
    groups = defaultdict(list)
    for r in records:
        groups[str(r["id"])].append(r)

    total_succ, total_cnt = 0, 0
    per_id = {}

    count = 0
    for gid, items in groups.items():
        if len(items) < require_min_group:
            continue
        refs  = [it.get("reference", "") for it in items]
        preds = [it.get("prediction", "") for it in items]
        n = min(len(refs), len(preds))
        if n == 0:
            per_id[gid] = 0.0
            continue

        succ = 0
        for i in range(n):
            gi = refs[i]
            # score gi vs ALL predictions
            row_scores = [scorer.score(gi, pj) for pj in preds]
            ranked = sorted(range(len(row_scores)), key=lambda j: -row_scores[j])
            true_j = i if strict_align else (i % len(preds))
            if true_j in ranked[:k]:
                succ += 1

        per_id[gid] = succ / n
        total_succ += succ
        total_cnt  += n

        count += 1
    print('total groups evaluated:', count)
    overall = total_succ / total_cnt if total_cnt > 0 else 0.0
    return overall, per_id


# Example:
file = 'path/to/your/predicted/caption/json/file'
with open(file, 'r') as f:
    records = json.load(f)

all_acc = []

for score in ['bleu4', 'meteor', 'rougeL', 'cider']:
    overall_acc, per_id = topk_classification_accuracy_coco(
        records,
        k=1,
        metric=score,
        require_min_group=2,
        strict_align=True
    )
    all_acc.append((score, overall_acc))
    print(f"Top-1 accuracy ({score}):", overall_acc)

print("Mean accuracy:", np.mean([acc for _, acc in all_acc]))