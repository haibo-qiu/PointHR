_base_ = ['../_base_/default_runtime.py',
          '../_base_/tests/segmentation.py']
# misc custom setting
batch_size = 12  # bs: total bs in all gpus
mix_prob = 0.8
empty_cache = False
enable_amp = True
seed = 58035118

# model settings
model = dict(
    type="pointhr_semseg",
    in_channels=6,
    num_classes=13,
    patch_embed_depth=1,
    patch_embed_channels=32,
    patch_embed_groups=4,
    patch_embed_neighbours=16,
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
    grid_sizes=(0.1, 0.2, 0.4, 0.8),
    attn_qkv_bias=True,
    pe_multiplier=False,
    pe_bias=True,
    attn_drop_rate=0.,
    drop_path_rate=0.3,
    enable_checkpoint=False,
    unpool_backend="map",  # map / interp
)

# scheduler settings
epoch = 3000
optimizer = dict(type='AdamW', lr=0.006, weight_decay=0.05)
scheduler = dict(type='MultiStepLR', milestones=[0.6, 0.8], gamma=0.1)

# dataset settings
dataset_type = "S3DISDataset"
data_root = "data/s3dis"

data = dict(
    num_classes=13,
    ignore_label=255,
    names=['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
           'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter'],
    train=dict(
        type='S3DISDataset',
        split=('Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6'),
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis='z', p=0.75),
            # dict(type="RandomRotate", angle=[-1, 1], axis='z', center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis='x', p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis='y', p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(type="Voxelize", voxel_size=0.04, hash_type='fnv', mode='train',
                 keys=("coord", "color", "label"), return_discrete_coord=True),
            dict(type="SphereCrop", point_max=100000, mode='random'),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            # dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "discrete_coord", "label"), feat_keys=["coord", "color"])
        ],
        test_mode=False
    ),
    val=dict(
        type='S3DISDataset',
        split='Area_5',
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"coord": "origin_coord", "label": "origin_label"}),
            dict(type="Voxelize", voxel_size=0.04, hash_type='fnv', mode='train',
                 keys=("coord", "color", "label"), return_discrete_coord=True),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect",
                 keys=("coord", "discrete_coord", "label"),
                 offset_keys_dict=dict(offset="coord"),
                 feat_keys=["coord", "color"])
        ],
        test_mode=False),
    test=dict(
        type='S3DISDataset',
        split='Area_5',
        data_root=data_root,
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(type='NormalizeColor')
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='Voxelize',
                voxel_size=0.04,
                hash_type='fnv',
                mode='test',
                keys=('coord', 'color'),
                return_discrete_coord=True),
            crop=None,
            post_transform=[
                dict(type='CenterShift', apply_z=False),
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'discrete_coord', 'index'),
                    feat_keys=('coord', 'color'))
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
                [dict(type="RandomScale", scale=[0.9, 0.9]),
                 dict(type="RandomFlip", p=1)],
                [dict(type="RandomScale", scale=[0.95, 0.95]),
                 dict(type="RandomFlip", p=1)],
                [dict(type="RandomScale", scale=[1, 1]),
                 dict(type="RandomFlip", p=1)],
                [dict(type="RandomScale", scale=[1.05, 1.05]),
                 dict(type="RandomFlip", p=1)],
                [dict(type="RandomScale", scale=[1.1, 1.1]),
                 dict(type="RandomFlip", p=1)],
            ]
        )
    )
)

criteria = [
    dict(type="CrossEntropyLoss",
         loss_weight=1.0,
         ignore_index=data["ignore_label"])
]
