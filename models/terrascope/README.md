---
license: apache-2.0
language:
  - zh
  - en
pipeline_tag: image-text-to-text
base_model: openbmb/MiniCPM-V-4.6
tags:
  - remote-sensing
  - vision-language
  - visual-grounding
  - object-counting
---

# TerraScope

> A compact vision-language model for remote-sensing imagery.

TerraScope is a 1.3B vision-language model adapted from MiniCPM-V 4.6 for aerial and satellite image understanding. It accepts one or two images with a natural-language query and is provided as merged standalone BF16 weights. No external LoRA adapter or application framework is required.

Remote-sensing adaptation by [Hengtao Wu](https://github.com/EternalWavee).

Model release: [LazySheep006/TerraScope-1.3B](https://modelscope.cn/models/LazySheep006/TerraScope-1.3B)

This release is the merged final checkpoint after supervised adaptation on public remote-sensing benchmarks and task-oriented instruction data. Training covers scene understanding, captioning, visual question answering, object counting, visual grounding, bi-temporal change understanding, and semantic-segmentation routing.

## Download

Download the complete model repository from ModelScope:

```bash
python -m pip install modelscope
modelscope download \
  --model LazySheep006/TerraScope-1.3B \
  --local_dir ./models/terrascope
```

An offline mirror is also maintained for network-isolated deployments. See the project-level [Model Weights](../../README.md#2-model-weights) instructions for the mirror link and expected directory layout. Keep all configuration, processor, tokenizer, and weight files together when redistributing the model.

## Model Details

| Property | Value |
| --- | --- |
| Architecture | MiniCPM-V 4.6 |
| Parameters | 1.3B |
| Modalities | Image and text to text |
| Languages | Chinese and English |
| Image input | Single-image and bi-temporal pairs |
| Precision | BF16 |
| Visual compression | 4x and 16x |
| Grounding coordinates | Normalized to `[0, 1000]` |
| Weight format | Merged standalone Safetensors |

## Capabilities

- Remote-sensing scene classification
- Detailed image description
- Object counting, including dense scenes
- Referring-expression grounding
- Bi-temporal change understanding
- Semantic-segmentation routing with `<seg>` and `<seg_all>`

The model is trained for direct natural-language interaction. Task prefixes are not required.

## Inference

```bash
pip install "transformers>=5.7.0" accelerate pillow torch
```

```python
import torch
from transformers import pipeline

pipe = pipeline(
    "image-text-to-text",
    model=".",
    dtype=torch.bfloat16,
    device_map="auto",
)

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "/path/to/image.jpg"},
        {"type": "text", "text": "How many buildings are in the image?"},
    ],
}]

result = pipe(
    text=messages,
    max_new_tokens=256,
    return_full_text=False,
)
print(result[0]["generated_text"])
```

Grounding responses follow:

```text
<ref>target</ref><box>x1 y1 x2 y2</box>
```

`<seg>` requests class-specific segmentation and `<seg_all>` requests full-scene semantic segmentation. These are routing tokens; pixel masks require a compatible downstream segmentation model.

## Intended Use

TerraScope is intended for remote-sensing research, education, demonstrations, and assisted image analysis. Performance can vary with sensor characteristics, geographic region, resolution, target scale, image compression, and prompt wording. Outputs should be independently verified before consequential use.

## License and Acknowledgements

TerraScope is derived from [MiniCPM-V 4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6), developed by the OpenBMB community and released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Redistribution should preserve all applicable upstream notices. Dataset and third-party component licenses remain separately applicable.

Upstream repository: [OpenBMB/MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V)
