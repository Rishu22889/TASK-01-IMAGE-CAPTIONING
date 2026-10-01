# Remote-Sensing Image Captioning on VRSBench

Fine-tuning **Qwen2.5-VL-3B-Instruct** with LoRA to caption satellite/aerial images from the open-source
[VRSBench](https://huggingface.co/datasets/xiang709/VRSBench) dataset, with the emphasis on **training-time optimization**:
what was changed, why, and what effect was measured.

Everything runs as Jupyter notebooks on a free Kaggle instance with **2× NVIDIA T4 (16 GB)**.

- Technical report: [`docs/technical_report.pdf`](docs/technical_report.pdf) (source: [`docs/technical_report.md`](docs/technical_report.md))
- Presentation: [`docs/presentation.pptx`](docs/presentation.pptx)

## Results at a glance

Final comparison on **1,000 held-out test images** (never used for any decision), primary metric **BERT-BLEU₄** (0–1, higher is better).
Δ is paired against E3 greedy, with a 95% bootstrap confidence interval.

| Model | Decoding | BERT-BLEU₄ | Δ vs E3 (95% CI) | BLEU-4 | Words (ref 48.4) | Inference s/image |
|---|---|---|---|---|---|---|
| E3: LoRA, 2k images, 1× T4 | greedy | 0.6694 | n/a | 0.097 | 46.0 | 1.07 |
| E8: LoRA, full data, 2× T4 DDP | greedy | 0.6767 | **+0.0073** [+0.0015, +0.0129] | **0.110** | 45.4 | 1.09 |
| **E8** | **MBR (6 candidates)** | **0.6860** | **+0.0166** [+0.0114, +0.0220] | 0.105 | 47.1 | 4.73 |

| Optimization | Measured effect |
|---|---|
| LoRA instead of full fine-tuning | 7.4–11.3 M trainable parameters (0.2–0.3%); peak memory 10.8–14.3 GB on a 16 GB T4 instead of ~60 GB |
| Vision-encoder LoRA, all 32 blocks, ViT LR 5× lower | best of the 2k-image runs (BERT-BLEU₄ 0.6586 → 0.6659), +41% time |
| Higher image resolution (324 instead of 256 visual tokens) | 2.4× training time, +3.4 GB, no quality gain → rejected |
| fp16 loss scaling (GradScaler) | earlier fp16 runs crashed with a non-finite loss; the full run finished with 4 of 2,533 steps skipped |
| DDP on 2× T4 + gradient accumulation (`no_sync`) | 2.49 img/s vs 1.22 img/s on one GPU (≈2×, near-linear); 20,264 images in 136 min |
| Full training split (20k instead of 2k images) | +0.0073 on test; BLEU-4 +14% relative |
| MBR decoding | +0.0093 over greedy on the same model, at 4.4× inference cost |

## Repository structure

```
├── README.md
├── requirements.txt
├── notebooks/
│   ├── E0_baseline.ipynb               LoRA on LLM attention, 2k images (baseline)
│   ├── E1_resolution_gc.ipynb          + higher image resolution + gradient checkpointing
│   ├── E2_vision_lora_last8.ipynb      + LoRA on the last 8 ViT blocks, differential LR
│   ├── E3_vision_lora_all32.ipynb      + LoRA on all 32 ViT blocks (best 2k-image run)
│   ├── E4_lr_scheduler.ipynb           E3 + linear warmup/decay schedule
│   ├── E5_grad_accumulation.ipynb      E3 + gradient accumulation 2
│   ├── E8_full_data_ddp.ipynb          final recipe: full data, 2× T4 DDP, GradScaler, warmup + cosine
│   ├── eval_decoding_bertbleu4.ipynb   decoding study, data-scaling curve, final 1,000-image test
│   └── compare_experiments.ipynb       BERT-BLEU₄ comparison of all runs with bootstrap CIs
├── results/
│   ├── experiments/<run>/              metrics.json + predictions.json (200 val images) per run
│   ├── comparison/                     tables and figures from compare_experiments.ipynb
│   └── decoding_eval/                  tables, figures, timings and all generated captions (val + test)
├── adapters/
│   ├── E3/                             LoRA adapter of the best 2k-image run
│   └── E8/                             LoRA adapter of the final model
└── docs/
    ├── technical_report.pdf
    ├── technical_report.md
    └── presentation.pptx
```

## Setup

**Hardware used:** Kaggle notebook, accelerator *GPU T4 ×2* (2× 16 GB), 4 vCPU, 30 GB RAM. Internet must be **ON**:
the base model and the dataset are downloaded from the Hugging Face Hub inside the notebooks.

**Dependencies:** the first cell of every notebook installs them. To set up a local environment instead:

```bash
pip install -r requirements.txt
```

`torchvision` must match the installed `torch` version (e.g. torch 2.10 ↔ torchvision 0.25), otherwise the Qwen image processor
fails to import. The runs used fp16 throughout, because the T4 has no bf16 support.

## How to reproduce

The dataset (`VRSBench_train.json`, `VRSBench_EVAL_Cap.json`, image archives) and the base model are downloaded automatically
by the notebooks. Paths are set in the first configuration cell of each notebook.

| Step | Notebook | GPU | Time (2× T4) | Output |
|---|---|---|---|---|
| 1 | `E0` … `E5` | 1× T4 | 20–47 min each | LoRA adapter + `metrics.json` + `predictions.json` per run |
| 2 | `E8_full_data_ddp.ipynb` | 2× T4 | ~2.3 h (+ ~3 min smoke test) | adapters at 25/50/75/100% of training, `metrics_E8.json`, `train_log.json` |
| 3 | `eval_decoding_bertbleu4.ipynb` | 2× T4 | ~4 h from scratch | everything in `results/decoding_eval/` |
| 4 | `compare_experiments.ipynb` | CPU is enough | ~2 min | everything in `results/comparison/` |

Notes:

- `E8_full_data_ddp.ipynb` writes the training script with `%%writefile` and launches it with `torchrun --nproc_per_node=2`.
  Run it with **Save Version → Save & Run All** so the adapters are kept as notebook output.
- `eval_decoding_bertbleu4.ipynb` takes the adapter folders in its `ADAPTERS` dict (`adapters/E3`, `adapters/E8` from this
  repo work as-is). Generated captions are cached; setting `PREVIOUS_RESULTS` to an uploaded copy of `results/decoding_eval`
  reuses them, so only missing captions are generated.
- The 200 validation images are `random.Random(2026).sample(...)` of `VRSBench_EVAL_Cap.json`. Their ids are listed in
  `results/experiments/val_subset_200.json`. The 1,000 test images are sampled (seed 7) from the remaining 9,150 images.

## Evaluation metric: BERT-BLEU₄

$$P_n = \frac{1}{|R_n|}\sum_{r\in R_n}\max_{c\in C_n}\cos\big(E(c),E(r)\big),\qquad
\text{BERT-BLEU}_4 = LP\cdot\exp\Big(\tfrac14\sum_{n=1}^{4}\log P_n\Big),\qquad
LP=\exp\Big(-0.5\,\frac{|L_C-L_R|}{L_R}\Big)$$

$C_n$ and $R_n$ are the n-gram sets of the candidate and reference caption. Implementation choices (one place in each notebook):
`bert-base-uncased`, last layer, each n-gram encoded on its own and mean-pooled over its word-pieces, lower-cased word tokens for
n-grams and lengths, score computed per caption and averaged. BLEU-4 (NLTK corpus BLEU) and BERTScore F1 are also reported, on
the 0–1 scale.

## Experiments (200 validation images, greedy decoding)

| Run | Change | Trainable params | BERT-BLEU₄ [95% CI] | Length penalty | BLEU-4 | Train time | Peak GPU memory |
|---|---|---|---|---|---|---|---|
| E0 | LoRA r=16 on LLM q/k/v/o, 2k images, 1 epoch, batch 1 | 7.37 M | 0.6586 [0.644, 0.672] | 0.846 | 0.088 | 19.5 min | 10.8 GB |
| E1 | + 324 visual tokens + gradient checkpointing | 7.37 M | 0.6541 [0.638, 0.670] | 0.837 | 0.089 | 47.3 min | 14.3 GB |
| E2 | + ViT LoRA (blocks 24–31), LR 1e-5 ViT / 5e-5 LLM | 8.36 M | 0.6625 [0.648, 0.676] | 0.851 | 0.096 | 21.3 min | 11.1 GB |
| E3 | + ViT LoRA (all 32 blocks) | 11.30 M | 0.6659 [0.652, 0.678] | 0.856 | 0.098 | 27.4 min | 13.1 GB |
| E4 | E3 + linear warmup/decay | 11.30 M | 0.6600 [0.644, 0.675] | 0.843 | 0.094 | 23.2 min* | 13.7 GB |
| E5 | E3 + gradient accumulation 2 | 11.30 M | 0.6607 [0.646, 0.675] | 0.844 | 0.091 | 27.0 min | 13.3 GB |
| E8 | E3 + full data (20,264), effective batch 8, warmup + cosine, GradScaler, 2× T4 DDP | 11.30 M | **0.6700** [0.656, 0.683] | 0.854 | **0.112** | 135.8 min (2 GPUs) | 12.7 GB / GPU |

\* E4 stopped at step 1,690 of 2,000 on a non-finite fp16 loss. A 2-epoch run (E6) failed the same way, and a LoRA-rank-32 run
(E7) gave no consistent gain; their notebooks are not part of this repository. On 200 images no run is significantly above
E0; the 1,000-image test above is what separates E3 from E8.

## Key observations

1. **The length penalty is the main loss.** Every model loses ~15% of its score to caption-length mismatch (mean LP ≈ 0.85),
   while the content term differs by < 0.01 between runs. Predicted length correlates only weakly with reference length (r ≈ 0.2–0.3).
2. **Resolution is not the bottleneck.** 324 instead of 256 visual tokens cost 2.4× training time and gave no gain.
3. **Adapting the vision encoder helps** (overhead images are far from the pretraining distribution), provided its LR stays low.
4. **fp16 without loss scaling is unstable on T4** (E4, E6 crashed); GradScaler made the 20k-image run finish.
5. **More data improved content and phrasing** (BLEU-4 +14%) but not length; on val the score plateaued after ~5k images.
6. **Beam search raises BLEU-4 but lowers BERT-BLEU₄** (shorter, more generic captions). MBR, which picks the consensus caption
   among 6 candidates, is the only decoding that improved BERT-BLEU₄.

## Known issues

- The E0–E5 notebooks saved predictions with the wrong `image_id` (they zipped `val_records` instead of `val_subset`);
  reference/prediction pairs were correct. The files in `results/experiments/` carry the corrected ids.
- Batched fp16 generation with left padding produced a broken caption (`!!!!` or random tokens) in 9 of the ~5,800 generated
  captions (5 distinct images); the original one-image-at-a-time E0–E5 evaluation had none. The test deltas hold without them: +0.0067 [+0.0015, +0.0119] (E8 vs E3) and +0.0086 [+0.0047, +0.0125]
  (MBR vs greedy) on the remaining 996 images. `eval_decoding_bertbleu4.ipynb` now regenerates such captions one image at a time
  (and with an fp32 vision encoder if needed). Its saved outputs come from the run before this step was added.
