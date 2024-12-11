# 配置继承，复用已有的通用配置内容，减少重复代码编写
_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

# 插件
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# 点云范围配置：点云数据在三维空间中的范围
# 用于限定模型处理的点云数据的空间边界，模型在训练和推理时会依据这个范围来筛选、处理相关数据点等操作
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
# 体素化时体素的大小: 点云数据进行体素化表示中，会根据这个尺寸来划分空间，将点云划分到不同的体素单元中
voxel_size = [0.15, 0.15, 4]

# 图像归一化的配置
# mean和std分别指定了图像3通道归一化时要减去的均值和除以的标准差, form ImageNet
# 将图像数据归一化到合适范围有助于模型训练和收敛
# to_rgb=True表示将图像数据转换为 RGB 格式，统一数据格式方便后续处理
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
# map has classes: divider, ped_crossing, boundary
map_classes = ['divider', 'ped_crossing','boundary']
# fixed_ptsnum_per_line = 20
# map_classes = ['divider',]
# ground truth 和 pred line 固定的点数，表达 line 使用的点数
fixed_ptsnum_per_gt_line = 20 # now only support fixed_pts > 0
fixed_ptsnum_per_pred_line = 20
eval_use_same_gt_sample_num_flag=True
num_map_classes = len(map_classes)

# 配置输入模态，指定模型输入数据来源
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

# 维度
_dim_ = 256             # 模型维度
_pos_dim_ = _dim_//2    # 位置编码维度
_ffn_dim_ = _dim_*2     # 前馈网络维度
_num_levels_ = 1        # Transformer层数
# bev_h_ = 50
# bev_w_ = 50
bev_h_ = 200            # BEV图像高度
bev_w_ = 100            # BEV图像宽度
queue_length = 1 # each sequence contains `queue_length` frames.

model = dict(
    type='MapTR',
    use_grid_mask=True,     # 格掩码，数据增强等，增强模型对不同区域图像特征的学习能力
    video_test_mode=False,
    pretrained=dict(img='ckpts/resnet50-19c8e357.pth'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='MapTRHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_vec=50,
        num_pts_per_vec=fixed_ptsnum_per_pred_line, # one bbox
        num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
        dir_interval=1,
        query_embed_type='instance_pts',
        transform_method='minmax',
        gt_shift_pts_pattern='v3',
        num_classes=num_map_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=2,
        code_weights=[1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='MapTRPerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=1,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='GeometrySptialCrossAttention',
                            pc_range=point_cloud_range,
                            attention=dict(
                                type='GeometryKernelAttention',
                                embed_dims=_dim_,
                                num_heads=4,
                                dilation=1,
                                kernel_size=(3,5),
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='MapTRDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],

                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='MapTRNMSFreeCoder',
            # post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
            pc_range=point_cloud_range,
            max_num=50,
            voxel_size=voxel_size,
            num_classes=num_map_classes),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.0),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_pts=dict(type='PtsL1Loss', 
                      loss_weight=5.0),
        loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.005)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='MapTRAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
            # reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            # iou_cost=dict(type='IoUCost', weight=1.0), # Fake cost. This is just to make it compatible with DETR head.
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
            pts_cost=dict(type='OrderedPtsL1Cost', 
                      weight=5),
            pc_range=point_cloud_range))))

# 数据集类型
dataset_type = 'CustomNuScenesLocalMapDataset'  # 继承
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

# 训练数据处理 pipeline
train_pipeline = [
    # 从文件中加载多视角图像数据，并将数据类型转换为float32
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    # 数据增强，改变图像的亮度、对比度、饱和度等光度属性
    dict(type='PhotoMetricDistortionMultiViewImage'),
    # 加载三维标注信息
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    # 依据前面定义的point_cloud_range对物体进行筛选，只保留在设定点云范围内的物体相关数据，去除范围外的数据
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # 按照class_names定义的类别列表对物体进行筛选，只保留属于这些指定类别的物体数据
    dict(type='ObjectNameFilter', classes=class_names),
    # 使用前面定义的img_norm_cfg配置对多视角图像进行归一化处理
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    # 对多视角图像进行随机缩放，缩放比例为0.5
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    # 对多视角图像进行填充操作，使得图像尺寸能被32整除
    dict(type='PadMultiViewImage', size_divisor=32),
    # 将处理后的三维数据（图像、标注等）整理成适合模型输入的默认格式
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    # 收集 key 对应的数据（gt_bboxes_3d、gt_labels_3d、img），形成最终要输入给模型进行训练的一批数据结构
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]

# 测试数据的处理 pipeline
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
   
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]

# 数据
data = dict(
    samples_per_gpu=4,  # batch size
    # workers_per_gpu=4,  # 每个 GPU 分配的用于数据加载等工作的进程数量，多线程
    workers_per_gpu=1,  # 每个 GPU 分配的用于数据加载等工作的进程数量，多线程
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        map_classes=map_classes,
        queue_length=queue_length,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
             map_ann_file=data_root + 'nuscenes_map_anns_val.json',
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             pc_range=point_cloud_range,
             fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
             eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
             padding_value=-10000,
             map_classes=map_classes,
             classes=class_names, modality=input_modality, samples_per_gpu=1),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
              map_ann_file=data_root + 'nuscenes_map_anns_val.json',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              pc_range=point_cloud_range,
              fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
              eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
              padding_value=-10000,
              map_classes=map_classes,
              classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

# 优化器
optimizer = dict(
    type='AdamW',
    lr=6e-4,    # 初始学习率
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),  # 学习率乘以0.1，使用 pretrained
        }),
    weight_decay=0.01)  # 权重衰减系数

# 优化器额外 config
# 梯度裁剪
# 当梯度的L2范数超过max_norm，梯度进行裁剪，将其范数限制在这个范围内，避免在训练过程中出现梯度爆炸的问题
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# 学习率调整策略
# learning policy
lr_config = dict(
    policy='CosineAnnealing',   # 余弦退火
    warmup='linear',            # 学习率预热方式
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)          # 学习率下降的最低下限比例，即学习率最低会降到初始学习率的1e-3倍

total_epochs = 110
# total_epochs = 50

# 模型评估：每2个训练轮次进行一次模型评估
# evaluation = dict(interval=1, pipeline=test_pipeline)
evaluation = dict(interval=2, pipeline=test_pipeline, metric='chamfer')

# runner
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

# 配置日志记录
# 每50次迭代记录一次日志信息
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# 混合精度训练
# 半精度 fp16占用2字节的内存空间，而单精度 fp32占用4字节
fp16 = dict(loss_scale=512.)

# 模型保存：每隔5个训练轮次保存一次模型
checkpoint_config = dict(interval=5)
