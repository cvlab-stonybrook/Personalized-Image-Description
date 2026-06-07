import os
import re
import sys
import json
import math
import torch
import spacy
import unicodedata

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Tuple, Callable, Dict, Optional
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration

import numpy as np
import matplotlib.pyplot as plt

# Add the path to cococaption to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cococaption')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pycocoevalcap')))

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from skimage import io



def to_int_id(x):
    """Turn string identifiers into deterministic integers."""
    try:
        return int(x)
    except Exception:
        return abs(hash(x)) % (10**8)


# --------------------------
# Simple noun extractor
# --------------------------

def extract_nouns(nlp, text: str):
    """
    Extract only NOUN / PROPN tokens in original order.
    Uses spaCy POS tagging.
    """
    doc = nlp(text)
    nouns = [t.lemma_.lower() for t in doc if t.pos_ in ("NOUN", "PROPN")]
    return nouns


def _is_synonym(a: str, b: str) -> bool:
    """Check if two words are synonyms using WordNet."""
    if a == b:
        return True
    syns_a = wn.synsets(a)
    syns_b = wn.synsets(b)
    # Skip if no synsets found
    if not syns_a or not syns_b:
        return False
    lemmas_a = {l.name().lower() for s in syns_a for l in s.lemmas()}
    lemmas_b = {l.name().lower() for s in syns_b for l in s.lemmas()}
    return len(lemmas_a & lemmas_b) > 0

def _is_stem_match(a: str, b: str) -> bool:
    """Check if stems are equal."""
    return stemmer.stem(a) == stemmer.stem(b)

def nw_matching(pred_words: list, ref_words: list, 
                gap_penalty: float = 0.0) -> float:
    """
    Ground-truth–normalized Needleman–Wunsch alignment for word-level matching
    that considers exact, stem, and synonym matches.
    """
    N, M = len(pred_words), len(ref_words)
    if N == 0 and M == 0:
        return 1.0
    if M == 0:
        return 0.0

    F = np.zeros((N + 1, M + 1), dtype=np.float32)
    for i in range(1, N + 1):
        F[i, 0] = F[i - 1, 0] + gap_penalty
    for j in range(1, M + 1):
        F[0, j] = F[0, j - 1] + gap_penalty

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            a, b = pred_words[i - 1].lower(), ref_words[j - 1].lower()
            
            # --- Multi-level matching ---
            if a == b:
                match = 1.0
            elif _is_stem_match(a, b):
                match = 0.8
            elif _is_synonym(a, b):
                match = 0.7
            else:
                match = 0.0

            F[i, j] = max(
                F[i - 1, j - 1] + match,  # match/mismatch
                F[i - 1, j] + gap_penalty, # deletion
                F[i, j - 1] + gap_penalty  # insertion
            )

    max_len = max(M, N)
    score = F[N, M] / max_len

    # # --- length penalty for longer predictions ---
    # length_ratio = N / (M + 1e-8)
    # if length_ratio > 1.0:
    #     penalty = length_penalty_weight * (length_ratio - 1.0)
    #     score -= penalty

    return float(np.clip(score, 0.0, 1.0))



# --------------------------
# Main evaluator
# --------------------------
def compute_sss(
        items: List[Dict],
        gap: float = 0.0,
        save_json: Optional[str] = None
    ) -> List[Dict]:
    """
    items: [{"id": ..., "prediction": ..., "reference": ...}, ...]
    """
    nlp = spacy.load("en_core_web_sm")

    # 1. extract nouns for each sample
    new_items = []
    for it_idx, it in enumerate(items):
        new_items.append(it)
    items = new_items

    pred_lists, ref_lists = [], []
    for it_idx, it in enumerate(items):
        pred_lists.append(extract_nouns(nlp, it["prediction"]))
        ref_lists.append(extract_nouns(nlp, it["reference"]))

    # 2. collect *all* nouns in order (with duplicates)
    all_nouns = []
    for lst in pred_lists + ref_lists:
        all_nouns.extend(lst)

    if len(all_nouns) == 0:
        # nothing to compare
        return [
            {"image_id": it["id"], "pred_nouns": [], "ref_nouns": [], "score": 1.0}
            for it in items
        ]

    # 5. per-sample similarity function using original vectors
    results = []
    for it, pred_w, ref_w in zip(items, pred_lists, ref_lists):
        score = nw_matching(pred_w, ref_w, gap_penalty=gap)
        results.append(
            {"image_id": it["id"], "subject": it["true_subject_id"], "pred_nouns": pred_w, "ref_nouns": ref_w, "score": score}
        )
    # with open(f"{save_json}_sss_results.json", "w", encoding="utf-8") as f:
    #     json.dump(results, f, ensure_ascii=False, indent=2)
    return results


import json

def find_top_k_sss(file_path):
    # NOTE: The file path is a placeholder. Please ensure this file exists and is accessible.
    TOP_K = 100 # Define the number of top results to print
    MIN_RECORDS = 2 # The minimum number of records required per image ID

    # Load data
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please check the path.")
        return
    except json.JSONDecodeError:
        print("Error: Could not decode JSON. Please check the file format.")
        return

    # 1. Aggregate all scores per image_id
    score_dict = {}
    for item in data:
        img_id = item.get("image_id")
        # if img_id is None:
        #     img_id = item.get("id")
        score = item.get("score")
        
        if img_id is None or score is None:
             continue 

        if img_id not in score_dict:
            score_dict[img_id] = []
        score_dict[img_id].append(score)

    # 2. Compute the average SSS score, but ONLY for images with >= MIN_RECORDS
    avg_scores = {}
    for img_id, scores in score_dict.items():
        # **This is the key filtering step:**
        if len(scores) >= MIN_RECORDS:
            avg_score = sum(scores) / len(scores)
            avg_scores[img_id] = avg_score

    # 3. Sort the images based on the average SSS score (highest first)
    sorted_scores = sorted(avg_scores.items(), key=lambda item: item[1], reverse=True)

    # 4. Print the top K image IDs with the highest SSS score
    print("-" * 70)
    print(f"Top {TOP_K} Image IDs with Highest Average SSS Score (Minimum {MIN_RECORDS} Records Required)")
    print("-" * 70)

    # Use min(TOP_K, len(sorted_scores)) to handle cases with fewer than TOP_K valid images
    top_n = min(TOP_K, len(sorted_scores))

    for i in range(top_n):
        img_id, avg_score = sorted_scores[i]
        # Get the original count of records for printing clarity
        record_count = len(score_dict[img_id])
        
        print(f"{i+1:3d}. Image ID: {img_id} (Records: {record_count}), Average SSS: {avg_score:.4f}")

# test_filtered_top_k() # Uncomment this line if you are running this as a standalone script


if __name__ == "__main__":
    stemmer = PorterStemmer()

    file = 'path/to/your/predicted/caption/json/file'
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    save_json = file.split('.')[0]
    results = compute_sss(data, save_json=save_json)
    avg_sss = sum([r["score"] for r in results]) / len(results)
    print(f"Average SSS: {avg_sss:.4f}")