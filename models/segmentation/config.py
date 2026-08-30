"""Inference-only SegFormer-B5 configuration for the LoveDA checkpoint."""

default_scope = "mmseg"

classes = (
    "background",
    "building",
    "road",
    "water",
    "barren",
    "forest",
    "agricultural",
)
palette = (
    (255, 255, 255),
    (255, 0, 0),
    (255, 255, 0),
    (0, 0, 255),
    (159, 129, 183),
    (0, 255, 0),
    (255, 195, 128),
)
metainfo = dict(classes=classes, palette=palette)

crop_size = (512, 512)
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)

model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type="MixVisionTransformer",
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 6, 40, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=None,
    ),
    decode_head=dict(
        type="SegformerHead",
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
    dict(type="PackSegInputs"),
]

test_dataloader = dict(
    dataset=dict(
        type="LoveDADataset",
        data_root="",
        data_prefix=dict(img_path=""),
        metainfo=metainfo,
        pipeline=test_pipeline,
        test_mode=True,
    )
)
