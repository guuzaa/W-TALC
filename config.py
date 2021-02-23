from yacs.config import CfgNode as CN

_C = CN()

_C.TRAIN = CN()
_C.TRAIN.lr = 0.0001
_C.TRAIN.batch_size = 10
_C.TRAIN.model_name = 'weakloc'
_C.TRAIN.pretrained_ckpt = None
_C.TRAIN.feature_size = 2048
_C.TRAIN.num_class = 20
_C.TRAIN.dataset_name = 'Thumos14reduced'
_C.TRAIN.max_seqlen = 750
_C.TRAIN._lambda = 0.5
_C.TRAIN.num_similar = 3
_C.TRAIN.seed = 1
_C.TRAIN.max_iter = 50000
_C.feature_type = 'I3D'



def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project"""
    # Returns a clone so that the defaults will not be altered.
    # This is for the "local variable" use pattern
    return _C.clone()
