_base_ = [
    '../_base_/models/hornet/hornet-tiny.py',
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

optim_wrapper = dict(optimizer=dict(lr=4e-3), clip_grad=dict(max_norm=100.0))

custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]

# load from which checkpoint
load_from = "https://download.openmmlab.com/mmclassification/v0/hornet/hornet-tiny_3rdparty_in1k_20220915-0e8eedff.pth"
