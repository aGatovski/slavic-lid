# Automatic Identification of Slavic Languages from Audio Using Deep Learning

*Faculty of Mathematics and Informatics, Sofia University · 2025–2026*

---

## Overview

We present a systematic study of automatic spoken language identification (LID) for eight Slavic languages spanning all three branches of the family: East (Belarusian, Russian, Ukrainian), West (Czech, Polish), and South (Bulgarian, Macedonian, Serbian). We design and evaluate a three-tier progression of deep learning systems of increasing model capacity, ranging from shallow convolutional networks trained from scratch to fine-tuned large pre-trained speech transformers. Across all tiers we explore different input representations (log-Mel spectrogram, MFCC with delta features, raw waveform), training objectives (cross-entropy, label smoothing, AAM-Softmax), and acoustic architectures (CNN, CNN+BiLSTM, Transformer encoder, XLS-R-300M, Whisper-small).

Our best system — a fine-tuned Whisper-small encoder with a two-layer classification head — achieves **macro-F1 of 0.795** on the in-domain test set and **0.817** on the out-of-domain FLEURS benchmark.

---

## Contributions

1. A comprehensive empirical study of three model tiers applied to 8-way Slavic LID from 3-second audio clips.
2. Systematic comparison of input representations, augmentation strategies (SpecAugment, Mixup), and training objectives (CE, label smoothing, AAM-Softmax).
3. Evaluation on both in-domain (MCV, 107k samples) and out-of-domain (FLEURS, 6k samples) benchmarks with full per-language analysis.
4. A finding that supervised multilingual pre-training (Whisper-small, 87M params) substantially outperforms a larger self-supervised model (XLS-R-300M, 300M+ params) — macro-F1 0.795 vs. 0.625.

---

## Dataset

**HuggingFace:** [`su-fmi-pytorch-slavic/slavic-languages-dataset`](https://huggingface.co/datasets/su-fmi-pytorch-slavic/slavic-languages-dataset) (gated — requires accepted access)

Sources: Mozilla Common Voice (MCV), VoxLingua107, Google FLEURS. All audio resampled to 16 kHz mono, clipped/padded to exactly 3 seconds.

| Split | Samples | Balance | Source |
|-------|---------|---------|--------|
| train | 80,000 | Balanced (10k/lang) | MCV + VoxLingua |
| test | 107,439 | Imbalanced (17.5×) | MCV only |
| FLEURS (OOD) | 6,157 | Balanced (~770/lang) | FLEURS only |

| Code | Language | Branch | Test samples |
|------|----------|--------|-------------|
| be | Belarusian | East | 29,232 |
| ru | Russian | East | 19,069 |
| uk | Ukrainian | East | 16,501 |
| cs | Czech | West | 14,441 |
| pl | Polish | West | 16,949 |
| bg | Bulgarian | South | 6,318 |
| mk | Macedonian | South | 3,263 |
| sr | Serbian | South | 1,666 |

> **Evaluation metric:** macro-F1 (unweighted mean of per-class F1). The 17.5× test imbalance makes micro-accuracy a misleading headline metric — it is reported for completeness only.

---

## Model Architectures

### Tier 1 — Shallow CNN
Four Conv2D blocks (widths 1→32→64→128→256, BatchNorm, ReLU, MaxPool) → Global Average Pooling → linear classifier. Evaluated with log-Mel (80 bins) and MFCC+Δ+ΔΔ (120-dim) inputs, with plain CE and label smoothing (ε=0.1).

### Tier 2 — Intermediate Acoustic Encoders
- **CNN + BiLSTM:** CNN frontend → 2-layer BiLSTM (hidden 256) → Attentive Statistics Pooling. 3.0M parameters.
- **CNN + Transformer:** CNN frontend → 4-layer pre-norm Transformer encoder (d=192, 4 heads, FFN 768) → Attentive Statistics Pooling. 2.3M parameters. Trained with AAM-Softmax (m=0.2, s=10).

### Tier 3 — Fine-Tuned Pre-Trained Models
- **XLS-R-300M:** Frozen convolutional feature extractor + fine-tuned 24-layer Transformer → mean-pool → CE or AAM-Softmax head. Layer-wise LR (encoder 1e-5, head 1e-3).
- **Whisper-small:** Positional embedding truncated to 150 tokens (3s input). Two-layer classification head (768→256→8). Two-phase fine-tuning: 4 epochs head-only, then 6 epochs full fine-tune.

---

## Results

### In-Domain Test Set (MCV · 107,439 samples)

| Tier | Configuration | Macro-F1 | Micro-Acc |
|------|--------------|----------|----------|
| 1 | CNN + Log-Mel + CE | 0.272 | 0.405 |
| 1 | CNN + Log-Mel + Label Smoothing | 0.246 | 0.328 |
| 1 | **CNN + MFCC+ΔΔ + CE** | **0.311** | **0.427** |
| 2 | CNN + BiLSTM + CE | 0.232 | 0.372 |
| 2 | CNN + BiLSTM + Mixup + CE | 0.238 | 0.378 |
| 2 | **CNN + Transformer + AAM-Softmax** | **0.288** | **0.425** |
| 3 | XLS-R-300M + CE | 0.480† | 0.609 |
| 3 | XLS-R-300M + AAM-Softmax | 0.625 | 0.691 |
| 3 | **Whisper-small + CE** | **0.795** | **0.848** |

*† Sub-optimal checkpoint; true peak estimated at 0.55–0.60.*

### Out-of-Domain FLEURS (6,157 samples)

| Tier | Configuration | Macro-F1 | Micro-Acc |
|------|--------------|----------|----------|
| 1 | CNN + Log-Mel + Label Smoothing | 0.166 | 0.190 |
| 2 | CNN + BiLSTM + Mixup + CE | 0.249 | 0.300 |
| 2 | CNN + Transformer + AAM-Softmax | 0.329 | 0.340 |
| 3 | XLS-R-300M + AAM-Softmax | 0.568 | 0.579 |
| 3 | **Whisper-small + CE** | **0.817** | **0.822** |

---

## Key Findings

**1. Pre-trained model selection is the dominant factor.**
Whisper-small (87M params, plain CE) at macro-F1 0.795 outperforms XLS-R-300M (300M+ params, AAM-Softmax) at 0.625 — supervised multilingual ASR pre-training on a task-relevant distribution matters more than model scale. Self-supervised pre-training (XLS-R) requires more fine-tuning signal to specialise for LID.

**2. From-scratch models are severely capacity-limited.**
All Tier 1 and Tier 2 configurations cluster between 0.23 and 0.31 macro-F1 regardless of loss, input features, or augmentation. The bottleneck is model capacity, not feature engineering — SpecAugment improves performance by only +0.008, and Mixup provides no measurable benefit.

**3. East Slavic (be/ru/uk) is the hardest sub-problem.**
Even at Tier 3 with XLS-R, 14.7% of Ukrainian samples are misclassified as Belarusian and 12.7% as Russian. These three languages share Cyrillic script, a common lexical core, and closely related phonological inventories; their differences are subtle in 3-second crowdsourced clips.

**4. Czech exhibits severe OOD collapse.**

| Model | Test F1 | FLEURS F1 | Drop |
|-------|---------|-----------|------|
| CNN + Transformer + AAM (Tier 2) | 0.410 | 0.113 | −0.297 |
| XLS-R + AAM-Softmax (Tier 3) | 0.690 | 0.223 | −0.467 |
| Whisper-small (Tier 3) | **0.908** | **0.744** | −0.164 |

Czech over-fits to MCV's crowdsourced acoustic profile; under weak models, out-of-domain Czech samples collapse into Serbian predictions — a cross-branch confusion that barely appears in-domain.

**5. Polish is the most robust language across all conditions.**
Polish nasal vowels and dense consonant clusters produce a highly distinctive acoustic signature that transfers across recording conditions. Polish achieves the best or near-best FLEURS F1 across all models (0.834 at Tier 3 C).

**6. South Slavic branch is consistently the hardest.**
Average branch F1 at Tier 3: East 0.838, West 0.898, South 0.685. All three South Slavic languages are minority test classes (together 10.4% of test), compounding the effect of within-branch linguistic similarity.

## References

- Radford et al. — [Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) (2022)
- Babu et al. — [XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale](https://arxiv.org/abs/2111.09296) (2021)
- Deng et al. — [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698) (2019)
- Park et al. — [SpecAugment](https://arxiv.org/abs/1904.08779) (2019)
- Conneau et al. — [FLEURS](https://arxiv.org/abs/2205.12446) (2022)
- Valk & Alumäe — [VoxLingua107](https://arxiv.org/abs/2011.12998) (2021)
