_base_ = [
    '../_base_/models/convnext/convnext-tiny.py',
    '../_base_/datasets/cifar10_bs128.py',
    '../_base_/schedules/cifar10_bs128_coslr_warmup_300e.py',
    '../_base_/default_runtime.py',
]

# Model settings
model = dict(
    head=dict(
        num_classes=10,
    ),
)

# schedule setting
optim_wrapper = dict(
    optimizer=dict(lr=4e-3),
    clip_grad=None,
)

# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (32 GPUs) x (128 samples per GPU)
auto_scale_lr = dict(base_batch_size=4096)

# load from which checkpoint
load_from = "https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_32xb128_in1k_20221207-998cf3e9.pth"
