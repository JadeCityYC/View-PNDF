"""
LLM report integration for View-PNDF.
=====================================

Consolidates the per-view reports produced by ``generate.py`` into a single
integrated radiology report, following the paper: view-specific findings are
generated independently and then merged by an LLM into one comprehensive report
(the object that NLG metrics in Tab. 1 / Tab. 2 are computed on).

Frontal and lateral predictions of the same study (matched by ``id``) are fed
to an OpenAI-compatible chat model with a merge prompt.  Studies with only one
available view fall back to that single view's report.

Credentials & endpoint are read from the environment / ``config.py`` -- no keys
are stored in this file.  Set:

    export OPENAI_API_KEY=...            # required
    export VIEWPNDF_LLM_BASE_URL=...     # optional (Azure / self-hosted gateway)
    export VIEWPNDF_INTEGRATOR_MODEL=... # optional (default: gpt-4o)

Output (``integrated.json``)::

    [ {"id": "...", "gt": "<frontal GT>", "integrated": "<merged report>"}, ... ]

Example
-------
    python integrate.py \
        --predictions ./outputs/iu_xray_vnf_predictions.json \
        --out ./outputs/iu_xray_vnf_integrated.json
"""

import os
import json
import argparse
from collections import defaultdict

import config

INTEGRATION_SYSTEM_PROMPT = (
    "You are an expert radiologist. You are given draft findings written "
    "independently from different chest X-ray views (e.g. frontal and lateral) "
    "of the SAME patient study. Consolidate them into a single, coherent, "
    "non-redundant radiology report. Resolve trivial wording differences, keep "
    "every distinct clinical finding, do not invent findings that are not "
    "supported by the drafts, and write in standard report style."
)


def build_user_prompt(view_reports):
    """view_reports: {view: text}."""
    parts = ["Here are the per-view draft findings:\n"]
    for view, text in view_reports.items():
        parts.append(f"[{view.upper()} VIEW]\n{text.strip()}\n")
    parts.append(
        "\nWrite the single consolidated report below. Output only the report text."
    )
    return "\n".join(parts)


def get_client():
    """Lazily build an OpenAI-compatible client from env/config."""
    from openai import OpenAI  # imported lazily so the rest of the repo has no hard dep

    api_key = os.environ.get(config.LLM_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"Set the {config.LLM_API_KEY_ENV} environment variable to your API key."
        )
    kwargs = {"api_key": api_key}
    if config.LLM_BASE_URL:
        kwargs["base_url"] = config.LLM_BASE_URL
    return OpenAI(**kwargs)


def integrate_one(client, model, view_reports, temperature=0.0, max_retries=3):
    """Merge one study's per-view reports; fall back to the single view if only
    one is present."""
    if len(view_reports) == 1:
        return next(iter(view_reports.values()))

    messages = [
        {"role": "system", "content": INTEGRATION_SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(view_reports)},
    ]
    last_err = None
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:  # noqa: BLE001 - transient API errors -> retry
            last_err = e
    raise RuntimeError(f"LLM integration failed after {max_retries} retries: {last_err}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True,
                        help="JSON from generate.py (per-view records)")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--model", type=str, default=config.LLM_INTEGRATOR_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gt_view", type=str, default="frontal",
                        help="which view's GT report to use as the reference for NLG scoring")
    args = parser.parse_args()

    with open(args.predictions) as f:
        records = json.load(f)

    # Group per-view predictions and GTs by study id.
    by_study = defaultdict(dict)      # id -> {view: pred}
    gt_by_study = defaultdict(dict)   # id -> {view: gt}
    for r in records:
        by_study[r["id"]][r["view"]] = r["pred"]
        gt_by_study[r["id"]][r["view"]] = r["gt"]

    client = get_client()

    out = []
    from tqdm import tqdm
    for study_id, view_reports in tqdm(by_study.items(), desc="integrating"):
        integrated = integrate_one(client, args.model, view_reports, temperature=args.temperature)
        gts = gt_by_study[study_id]
        gt = gts.get(args.gt_view) or next(iter(gts.values()))
        out.append({"id": study_id, "gt": gt, "integrated": integrated})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {len(out)} integrated reports -> {args.out}")


if __name__ == "__main__":
    main()
