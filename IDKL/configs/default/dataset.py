from yacs.config import CfgNode

dataset_cfg = CfgNode()

# config for dataset
dataset_cfg.sysu = CfgNode()
dataset_cfg.sysu.num_id = 395
dataset_cfg.sysu.num_cam = 6
dataset_cfg.sysu.data_root = "../dataset/SYSU-MM01"

dataset_cfg.regdb = CfgNode()
dataset_cfg.regdb.num_id = 206
dataset_cfg.regdb.num_cam = 2
dataset_cfg.regdb.data_root = "../dataset/RegDB"

dataset_cfg.llcm = CfgNode()
dataset_cfg.llcm.num_id = 713
dataset_cfg.llcm.num_cam = 2
dataset_cfg.llcm.data_root = "../dataset/LLCM"

