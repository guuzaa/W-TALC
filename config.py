from yacs.config import CfgNode as CN

_C = CN()

# default type:
# <class 'float'>, <class 'tuple'>, <class 'str'>, <class 'list'>, <class 'bool'>, <class 'int'>

_C.lr = 0.0001
_C.batch_size = 10
_C.model_name = 'weakloc'
_C.pretrained_ckpt = False
_C.feature_size = 2048
_C.num_class = 20
_C.dataset_name = 'Thumos14reduced'
_C.max_seqlen = 750
_C.Lambda = 0.5
_C.num_similar = 3
_C.seed = 1
_C.max_iter = 50000
_C.feature_type = 'I3D'
_C.print_every = 100


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project"""
    # Returns a clone so that the defaults will not be altered.
    # This is for the "local variable" use pattern
    return _C.clone()


if __name__ == '__main__':
    node = get_cfg_defaults()
    node.merge_from_file('experiment.yaml')
    print(node.dataset_name)
