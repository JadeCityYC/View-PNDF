"""
LLM-based clinical evaluation for View-PNDF (view-specific reports).
====================================================================

Traditional n-gram metrics cannot tell whether a view-specific report is
*clinically* faithful (paper §3.1): reports for different views can score
similarly while stating contradictory findings.  Following the paper we use an
LLM judge (e.g. GPT-4o; the paper also cross-checks with Qwen / DeepSeek) to
score each predicted report against its ground truth on a 1-10 scale for
semantic fidelity / diagnostic accuracy (paper Tab. 3 / Tab. 5).

The judge model and endpoint are configurable and the API key is read from the
environment -- nothing secret is stored here:

    export OPENAI_API_KEY=...
    export VIEWPNDF_LLM_BASE_URL=...     # optional (Azure / self-hosted vLLM, etc.)
    export VIEWPNDF_JUDGE_MODEL=gpt-4o   # or qwen2.5-72b-instruct, deepseek-chat, ...

Input is ``generate.py`` output (per-view ``pred`` + ``gt``).  We report the
mean score per view, so you can compare "w/o" vs "w/" View-PNDF (Tab. 3) or
different judges (Tab. 5).

Example
-------
    python evaluate_llm.py \
        --input ./outputs/iu_xray_vnf_predictions.json \
        --judge gpt-4o --out ./outputs/iu_xray_vnf_llm_scores.json
"""

import os
import re
import json
import argparse
from collections import defaultdict

import config

JUDGE_SYSTEM_PROMPT = (
    "You are an expert radiologist grading an automatically generated chest "
    "X-ray report against a reference (ground-truth) report. Judge how well the "
    "generated report captures the clinically significant findings of the "
    "reference: correct positive/negative findings, no fabricated pathology, and "
    "faithful description of the relevant anatomy for this view. Ignore stylistic "
    "differences."
)

JUDGE_USER_TEMPLATE = (
    "VIEW: {view}\n\n"
    "REFERENCE REPORT:\n{gt}\n\n"
    "GENERATED REPORT:\n{pred}\n\n"
    "Score the generated report from 1 (clinically wrong / contradictory) to 10 "
    "(clinically faithful to the reference). Respond with a JSON object exactly "
    'like {{"score": <int 1-10>, "reason": "<one short sentence>"}}.'
)


def get_client():
    from openai import OpenAI  # lazy import

    api_key = os.environ.get(config.LLM_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"Set the {config.LLM_API_KEY_ENV} environment variable to your API key."
        )
    kwargs = {"api_key": api_key}
    if config.LLM_BASE_URL:
        kwargs["base_url"] = config.LLM_BASE_URL
    return OpenAI(**kwargs)


def _parse_score(text):
    """Extract an integer 1-10 from the judge response (robust to extra prose)."""
    try:
        obj = json.loads(text)
        return int(obj["score"]), obj.get("reason", "")
    except Exception:  # noqa: BLE001 - fall back to regex
        m = re.search(r'"?score"?\s*[:=]\s*(\d+)', text) or re.search(r"\b([1-9]|10)\b", text)
        if m:
            return int(m.group(1)), text.strip()[:200]
    return None, text.strip()[:200]


def judge_one(client, model, rec, temperature=0.0, max_retries=3):
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
            view=rec["view"], gt=rec["gt"], pred=rec["pred"])},
    ]
    last_err = None
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
            )
            score, reason = _parse_score(resp.choices[0].message.content)
            if score is not None:
                return max(1, min(10, score)), reason
        except Exception as e:  # noqa: BLE001
            last_err = e
    print(f"[warn] judge failed for {rec.get('id')}: {last_err}")
    return None, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="generate.py per-view JSON")
    parser.add_argument("--judge", type=str, default=config.LLM_JUDGE_MODEL,
                        help="judge model id (gpt-4o / qwen / deepseek / ...)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_samples", type=int, default=-1, help="cap per view (cost control)")
    parser.add_argument("--out", type=str, default=None, help="optional per-sample scores JSON")
    args = parser.parse_args()

    with open(args.input) as f:
        records = json.load(f)

    client = get_client()

    from tqdm import tqdm
    scored = []
    per_view = defaultdict(int)
    for rec in tqdm(records, desc=f"judging ({args.judge})"):
        if 0 < args.max_samples <= per_view[rec["view"]]:
            continue
        score, reason = judge_one(client, args.judge, rec, temperature=args.temperature)
        if score is None:
            continue
        scored.append({**{k: rec[k] for k in ("id", "view")}, "score": score, "reason": reason})
        per_view[rec["view"]] += 1

    # Aggregate mean score per view.
    sums, counts = defaultdict(float), defaultdict(int)
    for s in scored:
        sums[s["view"]] += s["score"]
        counts[s["view"]] += 1

    print(f"\n=== LLM clinical scores (judge={args.judge}) ===")
    for view in sorted(counts):
        print(f"{view:8s}: {sums[view] / counts[view]:.2f}  (n={counts[view]})")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "judge": args.judge,
                "mean_by_view": {v: sums[v] / counts[v] for v in counts},
                "samples": scored,
            }, f, indent=2)
        print(f"wrote per-sample scores -> {args.out}")


if __name__ == "__main__":
    main()
