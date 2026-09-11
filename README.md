# View-PNDF

**View-specific Pattern Neuron Detection and Fine-tuning** for consistent
multi-view radiology report generation (RRG).

This repository implements the method of *"Seeing Through Multiple Views:
Parameter-Efficient Fine-Tuning via Selective Neurons for Consistent Radiology
Report Generation"*. Rather than fusing multi-view X-ray features, View-PNDF
locates the small subset of decoder neurons responsible for each view, verifies
them, and fine-tunes **only** those neurons — improving view-specific (esp.
lateral) reports while preserving general-view quality at a tiny parameter cost.

The pipeline has three method components plus generation/evaluation utilities:

| Stage | Script | What it does |
|-------|--------|--------------|
| **VND** — detection | `detect.py` | Collect per-neuron activation magnitude (Eq. 2) over view samples, keep top-γ per layer, intersect across samples (Eq. 3) → view pattern-neuron JSON. |
| **VNV** — verification | `deactivation.py` | Zero the detected neurons vs. an equal-size random control and compare generated reports (Eq. 4). |
| **VNF** — fine-tuning | `train.py` | Freeze the model; update **only** the detected pattern neurons via gradient masking (Eq. 5). |
| Generation | `generate.py` | Produce per-view reports from the base or a VNF checkpoint. |
| Integration | `integrate.py` | LLM-consolidate per-view reports into one integrated report. |
| NLG eval | `evaluate_nlg.py` | BLEU / METEOR / ROUGE-L on integrated reports. |
| LLM eval | `evaluate_llm.py` | LLM-as-judge (GPT-4o / Qwen / DeepSeek) clinical scoring of view-specific reports. |

## Setup

```bash
pip install -r requirements.txt   # core pipeline + evaluation/integration deps
cp .env.example .env              # then fill in your own paths / API key
```

All machine-specific values (dataset paths, model id, output dirs, LLM
endpoint) are **placeholders** in [`config.py`](config.py) and can be overridden
by env vars or CLI flags. No local paths or API keys are committed; the LLM API
key is read from `OPENAI_API_KEY` at call time.

### Data

Point `config.py` (or env vars) at your local copies:

- **IU-Xray** — a flat list of `{id, image_path:[...], report, view:[...]}`;
  re-split 70/10/20 with a fixed seed.
- **MIMIC-CXR** — a split dict `{"train":[...], "val":[...], "test":[...]}` with
  an image list and a view field (`view` / `view_position`). Splits are kept
  as-is.

Both are loaded through `data_rrg.load_dataset(name, ...)`.

## Usage

```bash
# 1. Detect view-specific pattern neurons
bash detect.sh                # or: python detect.py --dataset iu_xray ...

# 2. Verify them (qualitative deactivation vs. random control)
bash deactivation.sh

# 3. Fine-tune ONLY the detected neurons (multi-GPU DDP)
bash run.sh                   # or: python train.py --neuron_file ... --view lateral ...

# 4. Generate, integrate, and evaluate
bash generate.sh              # needs OPENAI_API_KEY for integrate/evaluate_llm
```

## Backbones

The reference config targets the public InternVL medical backbone (Phi-3
decoder, fused `qkv_proj` / `gate_up_proj`). To add another backbone
(MedGemma / LLaVA-Med / Hulu-Med), extend `view_pndf.build_projection_spec`
with the decoder's projection layout and, for split-projection families, add the
matching hook variant in `ActivationCollector`.

## Notes

- The pipeline attaches forward/backward hooks to the unmodified,
  officially-loaded model (no model-file rewrite; see the header of
  `view_pndf.py`), which keeps weight loading risk-free and architecture-agnostic.
- Evaluation scripts fall back to NLTK BLEU + pure-python ROUGE-L when
  `pycocoevalcap` is unavailable (METEOR is then skipped), so they run without
  a JRE, and print which backend was used.

## Reference

This work has been accepted to **MICCAI 2026**. If you find it useful, please
cite:

```bibtex
@inproceedings{chen2026viewpndf,
  title     = {Seeing Through Multiple Views: Parameter-Efficient Fine-Tuning
               via Selective Neurons for Consistent Radiology Report Generation},
  author    = {Chen, Yucheng and Zhu, Jinjing and Yu, Yang and Shi, Yufei and
               Naghshbandi, Hane and Liu, Jinhua and Koh, Angela S. and
               Fen, Fang and Ong, Kian Eng and Yeo, Si Yong},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year      = {2026},
}
```
