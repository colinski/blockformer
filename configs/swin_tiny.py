_base_ = ['./_base_/datasets/pipelines/rand_aug.py',
        './_base_/default_runtime.py'
]
# model settings
num_classes = 1000
window_size = (7, 7)
shift_size = (window_size[0]//2, window_size[1]//2)
Dh = 32 #head_dim
blocks = []
nH = 3 #num_heads
# blocks.append(dict(type='DownsampleBlock', in_channels=3, out_channels=nH*Dh, factor=4))
# win_cfg = dict(type='QKVAttention', qk_dim=nH*Dh, num_heads=nH)
# block = [
    # dict(type='WindowAttentionBlock', window_size=window_size, attn_cfg=win_cfg),
    # dict(type='FFNBlock', in_channels=nH*Dh, expansion_ratio=4, mode='x'),
    # dict(type='WindowAttentionBlock',window_size=window_size, shift_size=shift_size, attn_cfg=win_cfg),
    # dict(type='FFNBlock', in_channels=nH*Dh, expansion_ratio=4, mode='x'),
# ]
# for i in range(2): blocks.extend(block)

init_cfg = dict(type='Pretrained', checkpoint='/home/csamplawski_umass_edu/blockformer/checkpoints/resnet50_stem.pth')
blocks = [dict(type='ResnetStemBlock', init_cfg=init_cfg)]
blocks.append(dict(type='DownsampleBlock', in_channels=64, out_channels=nH*Dh, factor=1)) #not downsizing, just 64->96 channels
#output: B H/4 W/4 C

nH *= 2
win_cfg = dict(type='QKVAttention', qk_dim=nH*Dh, num_heads=nH)
blocks.append(dict(type='DownsampleBlock', in_channels=nH*Dh//2, out_channels=nH*Dh, factor=2))
block = [
    dict(type='WindowAttentionBlock', window_size=window_size, attn_cfg=win_cfg),
    dict(type='FFNBlock', in_channels=nH*Dh, expansion_ratio=4, mode='x'),
    dict(type='WindowAttentionBlock',window_size=window_size, shift_size=shift_size, attn_cfg=win_cfg),
    dict(type='FFNBlock', in_channels=nH*Dh, expansion_ratio=4, mode='x'),
]
for i in range(2): blocks.extend(block)
#output: B H/8 W/8 2C

nH *= 2
win_cfg = dict(type='QKVAttention', qk_dim=nH*Dh, num_heads=nH)
token_cfg = dict(type='QKVAttention', qk_dim=nH*Dh, num_heads=nH, seq_drop=0.0)
blocks.append(dict(type='DownsampleBlock', in_channels=nH*Dh//2, out_channels=nH*Dh, factor=2))
block = [
    dict(type='WindowAttentionBlock', window_size=window_size, attn_cfg=win_cfg),
    dict(type='FFNBlock', in_channels=nH*Dh, expansion_ratio=4, mode='x'),
    dict(type='WindowAttentionBlock',window_size=window_size, shift_size=shift_size, attn_cfg=win_cfg),
    dict(type='FFNBlock', in_channels=nH*Dh, expansion_ratio=4, mode='x'),
]
for i in range(6): blocks.extend(block)
# blocks.append(dict(type='SeqAttentionBlock', attn_cfg=token_cfg, mode='t'))
# blocks.append(dict(type='CrossAttentionBlock', attn_cfg=token_cfg))
# blocks.append(dict(type='FFNBlock', in_channels=nH*Dh, expansion_ratio=4, mode='t'))
#output: B H/16 W/16 4C

nH *= 2
win_cfg = dict(type='QKVAttention', qk_dim=nH*Dh, num_heads=nH)
token_cfg = dict(type='QKVAttention', qk_dim=nH*Dh, num_heads=nH, seq_drop=0.0)
blocks.append(dict(type='DownsampleBlock', in_channels=nH*Dh//2, out_channels=nH*Dh, factor=2))
block = [
    dict(type='WindowAttentionBlock', window_size=window_size, attn_cfg=win_cfg),
    dict(type='FFNBlock', in_channels=nH*Dh, expansion_ratio=4, mode='x'),
    dict(type='WindowAttentionBlock',window_size=window_size, shift_size=shift_size, attn_cfg=win_cfg),
    dict(type='FFNBlock', in_channels=nH*Dh, expansion_ratio=4, mode='x'),
]
for i in range(2): blocks.extend(block)
blocks.append(dict(type='SeqAttentionBlock', attn_cfg=token_cfg, mode='t'))
blocks.append(dict(type='CrossAttentionBlock', attn_cfg=token_cfg))
blocks.append(dict(type='FFNBlock', in_channels=nH*Dh, expansion_ratio=4, mode='t'))
blocks.append(dict(type='TokenPoolBlock'))





#output: B H/16 H/16 4C

#output: B H/32 H/32 8C

# blocks.append(dict(type='DownsampleBlock', in_channels=nH*Dh, out_channels=2*nH*Dh, factor=2))
# blocks.append(dict(type='SeqAttentionBlock', attn_cfg=token_cfg, mode='t'))
# blocks.append(dict(type='CrossAttentionBlock', attn_cfg=token_cfg))
# block.append(dict(type='FFNBlock', in_channels=nH*Dh, expansion_ratio=4, mode='t'))
#output: B H/16 H/16 4C

model = dict(type='ImageClassifier',
    backbone=dict(type='Blockbone', 
        blocks=blocks,
        num_tokens=100,
        token_dim=64,
    ),
    neck=None,
    head=dict(type='LinearClsHead',
        num_classes=num_classes,
        in_channels=768,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False
    ),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=num_classes, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=num_classes, prob=0.5)
    ])
)



# dataset settings
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies={{_base_.rand_increasing_policies}},
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in img_norm_cfg['mean'][::-1]],
            interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=img_norm_cfg['mean'][::-1],
        fill_std=img_norm_cfg['std'][::-1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        size=(256, -1),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

root = '/home/csamplawski_umass_edu'
data = dict(
    samples_per_gpu=50,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix=f'{root}/data/imagenet/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=f'{root}/data/imagenet/val',
        ann_file=f'{root}/data/imagenet/meta/val.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix=f'{root}/data/imagenet/val',
        ann_file=f'{root}/data/imagenet/meta/val.txt',
        pipeline=test_pipeline)
    )



paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        '.absolute_pos_embed': dict(decay_mult=0.0),
        '.relative_position_bias_table': dict(decay_mult=0.0)
    }
)

# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001
optimizer = dict(
    type='AdamW',
    lr=5e-4 * 128 * 8 / 512,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=paramwise_cfg
)
optimizer_config = dict(grad_clip=dict(max_norm=5.0))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-2,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=20 * 1252,
    warmup_by_epoch=False
)

                    
# optimizer
# optimizer_config = dict(grad_clip=None)
sampler = dict(type='RepeatAugSampler')
runner = dict(type='EpochBasedRunner', max_epochs=300)
evaluation = dict(interval=10, metric='accuracy')
find_unused_parameters=True

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

