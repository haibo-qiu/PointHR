#!/usr/bin/env python
# encoding: utf-8

_base_ = [
    '../_base_/datasets/shapenet_part.py',
    '../_base_/tests/part_segmentation.py',
    '../_base_/default_runtime.py'
]

batch_size = 24
batch_size_val = 24
batch_size_test = 8
eval_metric = "Ins_mIoU"
sync_bn = True
seed = 15817688

epochs = 100
start_epoch = 0
optimizer = dict(type='Adan', lr=0.01, weight_decay=0.01)
scheduler = dict(type='MultiStepLR', milestones=[0.6, 0.8], gamma=0.1)


model = dict(
    type="pointhrv2partv2",
    num_classes=50,
    num_shape_classes=16,
    in_channels=7,
    patch_embed_depth=1,
    patch_embed_channels=64,
    patch_embed_groups=8,
    patch_embed_neighbours=8,
    enc_depths=(1, 1, 5, 4),
    enc_blocks=(2, 2, 2, 2),
    enc_streams=(1, 2, 3, 4),
    enc_channels=(64, 32, 32, 32),
    enc_groups=(8, 4, 4, 4),
    enc_neighbours=(16, 16, 16, 16),
    dec_depths=(1, 1, 1, 1),
    dec_channels=(32, 32, 64, 128),
    dec_groups=(4, 4, 8, 16),
    dec_neighbours=(16, 16, 16, 16),
    grid_sizes=(0.0, 0.06, 0.12, 0.24),  # Gird Size
    attn_qkv_bias=True,
    pe_multiplier=False,
    pe_bias=True,
    attn_drop_rate=0.,
    drop_path_rate=0.3,
    unpool_backend="map",  # map / interp
)


# dataset settings
dataset_type = "ShapeNetPartDataset"
data_root = "data/shapenetpart"
cache_data = False
names = ["Airplane_{}".format(i) for i in range(4)] + \
        ["Bag_{}".format(i) for i in range(2)] + \
        ["Cap_{}".format(i) for i in range(2)] + \
        ["Car_{}".format(i) for i in range(4)] + \
        ["Chair_{}".format(i) for i in range(4)] + \
        ["Earphone_{}".format(i) for i in range(3)] + \
        ["Guitar_{}".format(i) for i in range(3)] + \
        ["Knife_{}".format(i) for i in range(2)] + \
        ["Lamp_{}".format(i) for i in range(4)] + \
        ["Laptop_{}".format(i) for i in range(2)] + \
        ["Motorbike_{}".format(i) for i in range(6)] + \
        ["Mug_{}".format(i) for i in range(2)] + \
        ["Pistol_{}".format(i) for i in range(3)] + \
        ["Rocket_{}".format(i) for i in range(3)] + \
        ["Skateboard_{}".format(i) for i in range(3)] + \
        ["Table_{}".format(i) for i in range(3)]


data = dict(
    num_classes=50,
    ignore_label=-1,  # dummy ignore
    names=names,
    train=dict(
        type=dataset_type,
        split=["train", "val"],
        data_root=data_root,
        transform=[
            dict(type="NormalizeCoord"),
            # dict(type="CenterShift", apply_z=True),
            # dict(type="RandomRotate", angle=[-1, 1], axis='z', center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 24, 1 / 24], axis='x', p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 24, 1 / 24], axis='y', p=0.5),
            dict(type="RandomScale", scale=[2/3, 3/2]),
            # dict(type="RandomFlip", p=0.5),
            # dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),

            # dict(type="Voxelize", voxel_size=0.01, hash_type='fnv', mode='train'),
            # dict(type="SphereCrop", point_max=2500, mode='random'),
            dict(type="ShufflePoint"),
            dict(type="UpdateHeight"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "cls_token", "label"), feat_keys=("coord", "norm", "height"))
        ],
        loop=2,
        test_mode=False,
    ),

    val=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="NormalizeCoord"),
            dict(type="UpdateHeight"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "cls_token", "label"), feat_keys=("coord", "norm", "height"))
        ],
        loop=1,
        test_mode=False,
        presample=True,
    ),

    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="NormalizeCoord"),
        ],
        loop=1,
        test_mode=True,
        presample=True,
        test_cfg=dict(
            post_transform=[
                dict(type="UpdateHeight"),
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "cls_token"), feat_keys=("coord", "norm", "height"))
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[2/3 + 0/6, 2/3 + 0/6])],
                [dict(type="RandomScale", scale=[2/3 + 1/6, 2/3 + 1/6])],
                [dict(type="RandomScale", scale=[2/3 + 2/6, 2/3 + 2/6])],
                [dict(type="RandomScale", scale=[2/3 + 3/6, 2/3 + 3/6])],
                [dict(type="RandomScale", scale=[2/3 + 4/6, 2/3 + 4/6])],
                [dict(type="RandomScale", scale=[2/3 + 5/6, 2/3 + 5/6])],
            ]
        )
    ),
)

criteria = [
    dict(type="SmoothCrossEntropy",
         loss_weight=1.0,
         ignore_index=data["ignore_label"],
         label_smoothing=0.2,
         num_classes=50)
]

