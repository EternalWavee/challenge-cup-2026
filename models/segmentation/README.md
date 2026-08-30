---
license: other
license_name: NVIDIA Source Code License for SegFormer
license_link: https://github.com/NVlabs/SegFormer/blob/master/LICENSE
pipeline_tag: image-segmentation
base_model: nvidia/mit-b5
datasets:
  - LoveDA
tags:
  - remote-sensing
  - semantic-segmentation
  - segformer
  - mmsegmentation
---

# TerraScope Segmentation

> A SegFormer-B5 model for semantic segmentation of remote-sensing imagery.

TerraScope Segmentation is a seven-class semantic-segmentation model based on SegFormer-B5 and fine-tuned on LoveDA. It produces dense, pixel-level predictions for common land-cover categories in urban and rural remote-sensing scenes. The model is provided as a standalone MMSegmentation checkpoint with its inference configuration.

LoveDA adaptation by [Ningkai Wu](https://github.com/kiny007).

Model release: [LazySheep006/TerraScope-Segformer](https://modelscope.cn/models/LazySheep006/TerraScope-Segformer)

## Download

Download the complete model repository from ModelScope:

```bash
python -m pip install modelscope
modelscope download \
  --model LazySheep006/TerraScope-Segformer \
  --local_dir ./models/segmentation
```

An offline mirror is also maintained for network-isolated deployments. See the project-level [Model Weights](../../README.md#2-model-weights) instructions for the mirror link and expected directory layout. Keep `config.py`, `checkpoint.pth`, the model card, and the accompanying license together when redistributing the model.

## Model Details

| Property | Value |
| --- | --- |
| Architecture | SegFormer-B5 |
| Backbone | MixVisionTransformer-B5 |
| Task | Semantic segmentation |
| Dataset | LoveDA |
| Number of classes | 7 |
| Inference framework | MMSegmentation 1.2.2 |
| Checkpoint format | PyTorch `.pth` |
| Best checkpoint | `best_mIoU_iter_10000.pth` |
| Validation mIoU | 53.85 |

## Classes

Class indices and palette entries follow the order below:

| Index | Class | Palette |
| ---: | --- | --- |
| 0 | Background | `(255, 255, 255)` |
| 1 | Building | `(255, 0, 0)` |
| 2 | Road | `(255, 255, 0)` |
| 3 | Water | `(0, 0, 255)` |
| 4 | Barren | `(159, 129, 183)` |
| 5 | Forest | `(0, 255, 0)` |
| 6 | Agricultural | `(255, 195, 128)` |

## Inference

The model directory contains both files required for standalone inference:

```text
config.py
checkpoint.pth
```

Install a PyTorch and MMCV build compatible with the target CUDA runtime, followed by MMSegmentation:

```bash
pip install mmengine==0.10.7 mmsegmentation==1.2.2
```

```python
from mmseg.apis import inference_model, init_model

model = init_model(
    "config.py",
    "checkpoint.pth",
    device="cuda:0",
)

result = inference_model(model, "/path/to/image.png")
mask = result.pred_sem_seg.data.squeeze(0).cpu().numpy()
print(mask.shape)
```

Each value in `mask` is a class index from the table above. The accompanying configuration applies RGB normalization and preserves the declared class order and palette.

## Intended Use

TerraScope Segmentation is intended for non-commercial research and evaluation in remote-sensing image analysis. Performance can vary with sensor characteristics, geographic region, ground sampling distance, atmospheric conditions, image compression, and domain shift. Predictions should be independently verified before consequential use.

## License and Acknowledgements

The original SegFormer project is distributed under the [NVIDIA Source Code License for SegFormer](https://github.com/NVlabs/SegFormer/blob/master/LICENSE), which restricts the work and its derivatives to non-commercial research or evaluation and requires the complete license to accompany redistribution. A copy is included with the model as [`LICENSE`](LICENSE).

The OpenMMLab MMSegmentation implementation is separately released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). LoveDA and other third-party assets remain subject to their respective licenses and terms. The NVIDIA license remains the controlling restriction wherever the SegFormer work or its derivatives apply.

Upstream repositories: [NVlabs/SegFormer](https://github.com/NVlabs/SegFormer) · [open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
