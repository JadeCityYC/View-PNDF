"""
NLG evaluation for View-PNDF (integrated reports).
===================================================

Computes BLEU-1..4, METEOR, and ROUGE-L for a set of (prediction, ground-truth)
report pairs -- the traditional metrics reported in the paper's Tab. 1 / Tab. 2
for the LLM-integrated reports.

Accepts either:
  * ``integrate.py`` output  (fields: ``integrated`` + ``gt``), or
  * ``generate.py`` output   (fields: ``pred`` + ``gt``), optionally filtered by
    ``--view`` for a per-view NLG breakdown (paper Tab. 4).

Metrics use the ``pycocoevalcap`` implementations (BLEU, METEOR, ROUGE) that are
standard in RRG papers.  Install with ``pip install pycocoevalcap`` (pulls in
the Java-based METEOR/CIDEr scorers).  If it is unavailable we fall back to a
pure-python ROUGE-L + NLTK BLEU so the script still runs, and say so.

Example
-------
    python evaluate_nlg.py --input ./outputs/iu_xray_vnf_integrated.json
    python evaluate_nlg.py --input ./outputs/iu_xray_vnf_predictions.json --view lateral
"""

import re
import json
import argparse


def _norm(text):
    """Light normalisation: lowercase, collapse whitespace, strip."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def load_pairs(path, pred_field=None, view=None):
    """Return (gts, preds) as parallel lists of normalised strings."""
    with open(path) as f:
        data = json.load(f)

    def pick_pred(rec):
        if pred_field:
            return rec[pred_field]
        return rec.get("integrated", rec.get("pred"))

    gts, preds = [], []
    for rec in data:
        if view is not None and rec.get("view") != view:
            continue
        gts.append(_norm(rec["gt"]))
        preds.append(_norm(pick_pred(rec)))
    return gts, preds


# --------------------------------------------------------------------------- #
# Preferred backend: pycocoevalcap (matches RRG-paper convention)
# --------------------------------------------------------------------------- #
def _score_coco(gts, preds):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge

    gts_d = {i: [g] for i, g in enumerate(gts)}
    res_d = {i: [p] for i, p in enumerate(preds)}

    scores = {}
    bleu, _ = Bleu(4).compute_score(gts_d, res_d)
    scores.update({f"BLEU-{i+1}": s for i, s in enumerate(bleu)})
    scores["METEOR"], _ = Meteor().compute_score(gts_d, res_d)
    scores["ROUGE-L"], _ = Rouge().compute_score(gts_d, res_d)
    return scores


# --------------------------------------------------------------------------- #
# Fallback backend: NLTK BLEU + pure-python ROUGE-L (no METEOR)
# --------------------------------------------------------------------------- #
def _rouge_l(gt, pred):
    a, b = gt.split(), pred.split()
    if not a or not b:
        return 0.0
    # LCS length via DP.
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(b), lcs / len(a)
    beta = 1.2
    return ((1 + beta ** 2) * prec * rec) / (rec + beta ** 2 * prec)


def _score_fallback(gts, preds):
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    refs = [[g.split()] for g in gts]
    hyps = [p.split() for p in preds]
    sm = SmoothingFunction().method1
    scores = {}
    for n in range(1, 5):
        w = tuple([1.0 / n] * n + [0.0] * (4 - n))
        scores[f"BLEU-{n}"] = corpus_bleu(refs, hyps, weights=w, smoothing_function=sm)
    scores["ROUGE-L"] = sum(_rouge_l(g, p) for g, p in zip(gts, preds)) / max(len(gts), 1)
    scores["METEOR"] = float("nan")  # not computed in fallback
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="integrate.py or generate.py JSON")
    parser.add_argument("--pred_field", type=str, default=None,
                        help="override which field holds the prediction (default: integrated|pred)")
    parser.add_argument("--view", type=str, default=None,
                        help="for generate.py output, restrict to one view (frontal/lateral)")
    args = parser.parse_args()

    gts, preds = load_pairs(args.input, args.pred_field, args.view)
    if not gts:
        raise SystemExit("No (gt, pred) pairs found -- check --input / --view.")
    print(f"scoring {len(gts)} report pairs"
          + (f" (view={args.view})" if args.view else ""))

    try:
        scores = _score_coco(gts, preds)
        backend = "pycocoevalcap"
    except Exception as e:  # noqa: BLE001
        print(f"[info] pycocoevalcap unavailable ({e}); using NLTK/pure-python fallback (no METEOR).")
        scores = _score_fallback(gts, preds)
        backend = "fallback"

    print(f"\n=== NLG metrics ({backend}) ===")
    for k in ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L"]:
        if k in scores:
            print(f"{k:8s}: {scores[k]:.4f}")


if __name__ == "__main__":
    main()
