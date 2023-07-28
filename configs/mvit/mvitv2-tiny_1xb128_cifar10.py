_base_ = [
    '../_base_/models/mvit/mvitv2-tiny.py',
    '../_base_/datasets/cifar10_bs128.py',
    '../_base_/schedules/cifar10_bs128_coslr_warmup_300e.py',
    '../_base_/default_runtime.py'
]

# Model settings
model = dict(
    head=dict(
        num_classes=10,
    ),
)

# schedule settings
optim_wrapper = dict(
    optimizer=dict(lr=2.5e-4),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.pos_embed': dict(decay_mult=0.0),
            '.rel_pos_h': dict(decay_mult=0.0),
            '.rel_pos_w': dict(decay_mult=0.0)
        }),
    clip_grad=dict(max_norm=1.0),
)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=2048)

# load from which checkpoint
load_from = "https://download.openmmlab.com/mmclassification/v0/mvit/mvitv2-tiny_3rdparty_in1k_20220722-db7beeef.pth"
