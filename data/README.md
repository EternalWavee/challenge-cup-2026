# TerraScope Training Corpus

This directory documents the supervised fine-tuning and evaluation splits used for TerraScope. It is included for reproducibility and is not required for inference.

## Splits

| File | Records | Role |
| --- | ---: | --- |
| `train.jsonl` | 285,154 | Supervised fine-tuning |
| `val_5k.jsonl` | 5,000 | Training-time validation |
| `val_full.jsonl` | 30,966 | Full validation split |
| `test.jsonl` | 34,913 | Held-out evaluation |

The corpus covers scene classification, image captioning, object counting, visual grounding, bi-temporal change understanding, and segmentation routing. Records use a unified system prompt and place `<image>` tokens at the beginning of the visual input.

## Paths

Image references are relative to `ROOT_IMAGE_DIR`:

```text
<ROOT_IMAGE_DIR>/
├── dataset/...
└── source/...
```

Set `ROOT_IMAGE_DIR` before training or batch evaluation. Questions, answers, coordinates, and annotations must not be rewritten during path relocation. Dataset splits should remain immutable for a given model version.

## Notes

- Keep training, validation, and test images disjoint where the source dataset permits.
- Preserve the licenses and attribution of each source dataset.
- Exclude qualitative answers such as `Multiple`, `多个`, `许多`, and `大量` from exact-count evaluation.
- The data manifests are not required for inference.
