# SLDPC

Official codebase for:

**SLDPC: Slide-Level Dual-Prompt Collaboration for Few-Shot Whole Slide Image Classification**

Published in *Computerized Medical Imaging and Graphics* (Elsevier), 2026.
[Article on ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0895611126000716)
&nbsp;·&nbsp; DOI: [10.1016/j.compmedimag.2026.102768](https://doi.org/10.1016/j.compmedimag.2026.102768)
&nbsp;·&nbsp; *In Press, Journal Pre-proof*

---

## What this repository provides

A reference implementation of SLDPC on the **TITAN** slide-level VLM backbone, including:

- **Stage-1**: Continuous Prompt Initialization (CPI) trained with cross-entropy.
- **Stage-2**: Dynamic Hard Negative Sampling (DHNO) + symmetric InfoNCE (SICL).
- **Inference**: Weighted fusion (WFM) of base prompt `P` and task-specific prompt `P'`.
- A **zero-shot baseline** integrated into the same pipeline (TITAN's 23-template ensemble), which reproduces the "TITAN" rows of paper Table 2.
- Training entrypoint, CLI, and dataset configs for **UBC-OCEAN, TCGA-NSCLC, TCGA-RCC, TCGA-OT**.

> **Note on PRISM.** The paper additionally reports SLDPC+PRISM results.
> The PRISM adapter is **not part of this release**. Only the TITAN code path is published here.

---

## Repository layout

```text
SLDPC/
├── sldpc/
│   ├── backbones/
│   │   ├── registry.py             # get_backbone(name="titan", ...)
│   │   └── titan/                  # PromptedTitan, TitanPromptLearner, encode_text
│   ├── core/
│   │   ├── prompt_learner_base.py  # P / P' / fused
│   │   ├── losses.py               # CE + symmetric InfoNCE
│   │   ├── negative_sampler.py     # DHNO
│   │   └── fusion.py               # WFM (omega-weighted ctx mix)
│   ├── data/
│   │   └── slide_feature_dataset.py
│   ├── trainers/
│   │   ├── base_trainer.py
│   │   ├── stage1_trainer.py
│   │   ├── stage2_trainer.py
│   │   └── titan_pipeline.py       # main entry — argparse + Stage1 + Stage2 + ZS
│   └── utils/
│       ├── metrics.py              # ACC / Macro-F1 / AUC
│       ├── seed.py
│       ├── run_logging.py
│       └── zero_shot.py
├── scripts/
│   └── train_titan.py              # thin CLI wrapper
├── configs/
│   ├── backbones/titan.yaml
│   └── datasets/
│       ├── ubc_ocean.yaml
│       ├── tcga_nsclc.yaml
│       ├── tcga_rcc.yaml
│       ├── tcga_ot.yaml
│       └── data_io_config.yaml     # CSV schema for split parsing
├── data/
│   └── datasets/
│       ├── ubc_ocean/              # CSV splits only (no features, no images)
│       ├── tcga_nsclc/
│       ├── tcga_rcc/
│       ├── tcga_ot/
│       └── zero_shot_prompts/      # per-dataset class synonyms (yaml)
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 1. Installation

> **Versions are pinned.** TITAN's `trust_remote_code` modeling files
> were written against `transformers==4.46.0` and `torch==2.0.1+cu117`,
> and the entire pinned stack in `requirements.txt` mirrors the
> environment that produced the paper's numbers. Please install
> exactly those versions — newer `transformers` (≥ 4.50) breaks
> TITAN's class hierarchy at load time, and other PyTorch versions
> may produce subtly different numbers.

`requirements.txt` installs the GPU build of PyTorch from PyTorch's
CUDA 11.7 wheel index. The cu117 wheels are forward-compatible with
newer NVIDIA drivers (CUDA 11.8 / 12.x), so this works on most
modern systems without changes.

**Recommended: conda + Python 3.9**

```bash
conda create -n sldpc python=3.9 -y
conda activate sldpc

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

**Alternative: venv (Linux / macOS)**

```bash
python3.9 -m venv .venv          # use Python 3.9, 3.10, or 3.11
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

**Alternative: venv (Windows PowerShell)**

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If PowerShell activation fails with *"running scripts is disabled on
this system"*, allow it for the current session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**Verify the GPU is picked up**

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

This should print `cuda: True`. If it prints `False`, your driver may
be too old for cu117, or you may not have an NVIDIA GPU — see the
"CPU-only" section below.

**Tested**: Python 3.9.22, PyTorch 2.0.1+cu117, transformers 4.46.0,
timm 1.0.3, einops 0.6.1.

**CPU-only / no NVIDIA GPU**

If you do not have an NVIDIA GPU (e.g. on a Mac, or a CPU-only Linux
box), the cu117 wheels will fail to resolve. Edit `requirements.txt`
and replace the GPU torch lines with their plain CPU equivalents:

```text
# Comment out or remove these:
# --index-url https://download.pytorch.org/whl/cu117
# --extra-index-url https://pypi.org/simple
# torch==2.0.1+cu117
# torchvision==0.15.2+cu117

# Add these instead:
torch==2.0.1
torchvision==0.15.2
```

Training on CPU works but is roughly 30–50× slower per epoch.

**Windows users**: clone or place this repository under a path that
contains only ASCII characters (no Chinese, Japanese, accented letters,
etc.). PyTorch 2.0.1 has a known Windows-specific issue
([pytorch#98918](https://github.com/pytorch/pytorch/issues/98918),
[pytorch#103949](https://github.com/pytorch/pytorch/issues/103949))
where `torch.save` / `torch.load` fail with confusing
*"Parent directory ... does not exist"* errors when the path contains
multi-byte characters. This codebase already opens checkpoints via file
handles to mitigate this, but the safest practice remains to keep the
project root ASCII-only. Spaces in the path are fine.

---

## 2. Authenticate for TITAN

TITAN weights are gated on Hugging Face and are **not redistributed by this
repository**. To run the code you must:

1. Request and accept access at <https://huggingface.co/MahmoodLab/TITAN>.
2. Authenticate locally:

   ```bash
   huggingface-cli login
   ```

If your environment is offline, point the CLI to a local snapshot:

```bash
python scripts/train_titan.py \
    --hf-model-id /path/to/local_titan_snapshot \
    --hf-local-files-only \
    ...
```

---

## 3. Prepare slide-level features

This codebase consumes **pre-extracted slide-level features**, not raw WSIs.
For each dataset you need a single `.pkl` file with two keys:

| Key          | Type                         | Description                                                  |
|--------------|------------------------------|--------------------------------------------------------------|
| `embeddings` | `np.ndarray (N, 768)`        | Slide-level features from TITAN's slide encoder              |
| `filenames`  | `list[str]` of length `N`    | Slide identifiers, matching the `slide_id` column of the CSV |

The recommended extraction pipeline matches what the paper used:

1. **Tile the WSI** at 20× magnification into non-overlapping 512×512
   patches inside tissue regions (HEST tissue/background segmentation).
2. **Patch features**: extract a 768-dim feature per tile using
   [CONCH](https://huggingface.co/MahmoodLab/CONCH).
3. **Slide features**: aggregate patch features through TITAN's slide
   encoder to get one 768-dim embedding per slide.
4. **Pack** all `(slide_id, embedding)` pairs into a single `.pkl` with
   the keys above.

The end-to-end **TRIDENT** pipeline (Zhang et al., 2025,
[arXiv:2502.06750](https://arxiv.org/abs/2502.06750)) automates steps 1–3.

> The dataset CSVs under `data/datasets/<name>/` give the slide-id
> universe and class labels for each benchmark, but **no feature files
> or images are redistributed** in this repository — TITAN/CONCH outputs
> are subject to the upstream licenses, and TCGA / UBC-OCEAN raw data
> are subject to their respective data-use agreements. You must extract
> the features yourself.

---

## 4. Run the main experiment

UBC-OCEAN, 16-shot, 5 random seeds, fixed split (matches paper Table 1):

```bash
python scripts/train_titan.py \
    --dataset ubc_ocean --few-shot-k 16 \
    --features-path /path/to/UBC-OCEAN_titan_features.pkl \
    --fixed-train-csv data/datasets/ubc_ocean/UBC-OCEAN_all_train.csv \
    --fixed-test-csv  data/datasets/ubc_ocean/UBC-OCEAN_all_test.csv \
    --n-seeds 5 \
    --topk 4 --omega 0.6 --best-metric F1 \
    --output-dir runs/main/ubc_ocean_k16
```

Outputs:

- Per seed: `runs/.../seed_<s>/final_report.json` with `zero_shot`,
  `stage1`, and `stage2` blocks.
- Aggregated: `runs/.../seed_summary.json` with mean ± std across seeds.

The defaults faithfully reproduce the legacy reference implementation
that generated the paper's numbers. CLI flags are available to switch
each algorithmic choice (loss, sampling strategy, eval mode, …) — run
`python scripts/train_titan.py --help` for the full list.

### Other datasets

```bash
# TCGA-NSCLC (2 classes)
python scripts/train_titan.py --dataset tcga_nsclc --few-shot-k 16 \
    --features-path /path/to/TCGA-NSCLC_titan_features.pkl \
    --fixed-train-csv data/datasets/tcga_nsclc/TCGA-NSCLC_train.csv \
    --fixed-test-csv  data/datasets/tcga_nsclc/TCGA-NSCLC_test.csv \
    --n-seeds 5 --topk 2 --omega 0.8 \
    --output-dir runs/main/tcga_nsclc_k16

# TCGA-RCC (3 classes)
python scripts/train_titan.py --dataset tcga_rcc --few-shot-k 16 \
    --features-path /path/to/TCGA-RCC_titan_features.pkl \
    --fixed-train-csv data/datasets/tcga_rcc/TCGA-RCC_train.csv \
    --fixed-test-csv  data/datasets/tcga_rcc/TCGA-RCC_test.csv \
    --n-seeds 5 --topk 3 --omega 0.8 \
    --output-dir runs/main/tcga_rcc_k16

# TCGA-OT (46 classes; pre-extracted slide features only)
python scripts/train_titan.py --dataset tcga_ot --few-shot-k 16 \
    --features-path /path/to/TCGA-OT_titan_features.pkl \
    --fixed-train-csv data/datasets/tcga_ot/tcga-ot_train.csv \
    --fixed-test-csv  data/datasets/tcga_ot/tcga-ot_test.csv \
    --n-seeds 5 --topk 4 --omega 0.8 \
    --output-dir runs/main/tcga_ot_k16
```

---

## 5. Ablations and switches

The most useful CLI knobs (paper Table 4 / Table 10 / Fig. 4):

| Flag                                           | Effect                                                            |
|------------------------------------------------|-------------------------------------------------------------------|
| `--skip-stage2`                                | Run Stage-1 only (Table 4 Baseline / CPI-only).                   |
| `--skip-stage1 --stage2-init random`           | Random `P'` init (Table 4 "w/o-CPI").                             |
| `--stage2-dhno-mode {full,sampling_only,none}` | DHNO ablation (Table 4).                                          |
| `--stage2-loss {symmetric,i2t,t2i,ce}`         | Loss-direction ablation (Table 10).                               |
| `--stage2-eval-mode {fused,task,base}`         | Inference prompt: `P̃`, `P'`, or `P` (Eq. 9 / Table 4 WFM).      |
| `--omega <float>`                              | Fusion weight ω in Eq. 9 (default 0.8 for most datasets).         |
| `--topk <int>`                                 | Hard-negative count K in DHNO (Table 7/8).                        |
| `--n-ctx <int>`                                | Learnable context length M (Table 5).                             |
| `--csc`                                        | Class-specific context (Table 6); default off (class-unified).    |
| `--skip-zero-shot`                             | Skip the TITAN zero-shot baseline (default: on).                  |

### Hardware and time

On a single NVIDIA RTX 3090 / 4090, a UBC-OCEAN 16-shot × 5-seed run
takes roughly 6–10 minutes end-to-end (TITAN backbone is frozen; only
~8.2K prompt parameters are trained). TCGA-OT (46 classes) is slower
due to the larger text head — budget ~20 minutes.

---

## 6. Citation

```bibtex
@article{Yuan2026SLDPC,
    author  = {Yuan, Lulin and Zheng, Yifeng and Liu, Weiqiang and Zhao, Hong and Zhang, Wenjie and Wei, Baoya and Chen, Liming},
    title   = {{SLDPC}: Slide-Level Dual-Prompt Collaboration for few-shot whole slide image classification},
    journal = {Computerized Medical Imaging and Graphics},
    year    = {2026},
    issn    = {0895-6111},
    doi     = {10.1016/j.compmedimag.2026.102768},
    url     = {https://www.sciencedirect.com/science/article/pii/S0895611126000716},
    note    = {In Press, Journal Pre-proof. Available online 21 April 2026.},
}
```

> **Note**: this entry corresponds to the *In Press, Journal Pre-proof*
> version. Update `volume`, `number`, and `pages` once the final version
> of record is assigned an issue.

If you build on the TITAN backbone, please also cite the upstream model:

> Ding, T. et al. *A multimodal whole-slide foundation model for pathology.*
> Nature Medicine, 2025.

---

## 7. License

Code in this repository is released under the [Apache License 2.0](LICENSE).

Third-party constraints that apply when you use this code:

- **TITAN weights** (`MahmoodLab/TITAN`): gated on Hugging Face under
  CC-BY-NC-ND 4.0. **Not redistributed here.** Obtain access from the
  upstream model page.
- **CONCH weights** (`MahmoodLab/CONCH`): gated on Hugging Face. Used by
  the upstream feature extraction pipeline, not by this repository
  directly.
- **TCGA / UBC-OCEAN data**: subject to their respective data-use
  agreements. Only public slide identifiers and labels are included
  in `data/datasets/`; no images, patches, or extracted features are
  redistributed.

If you redistribute features extracted with TITAN/CONCH, you are bound
by the upstream license terms.

---

## 8. Acknowledgements

This work builds on:

- **TITAN** (Ding et al., 2025) — slide-level multimodal foundation model.
- **CONCH** (Lu et al., 2024) — patch-level vision-language pretraining.
- **TRIDENT** (Zhang et al., 2025) — pathology data processing pipeline.
- **CoOp** / **DPC** — prompt-tuning families that inspired the
  dual-prompt design.

We thank the authors of these projects for releasing their code and
models to the community.
