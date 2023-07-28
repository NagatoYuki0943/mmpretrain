_base_ = [
    '../_base_/models/efficientnet_b0.py',
    '../_base_/datasets/cifar10_bs128.py',
    '../_base_/schedules/cifar10_bs128_coslr_warmup_300e.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    head=dict(
        num_classes=10,
    ),
)

# load from which checkpoint
load_from = "https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty-ra-noisystudent_in1k_20221103-75cd08d3.pth"
