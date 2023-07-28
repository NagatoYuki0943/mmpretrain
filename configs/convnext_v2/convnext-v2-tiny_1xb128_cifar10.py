_base_ = [
    '../_base_/models/convnext_v2/tiny.py',
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
    optimizer=dict(lr=3.2e-3),
    clip_grad=None,
)

# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]

# load from which checkpoint
load_from = "https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-tiny_3rdparty-fcmae_in1k_20230104-80513adc.pth"
