from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
from easydict import EasyDict as edict

config = edict()

config.WORKERS = 32
config.LOG_DIR = ''
config.TENSORBOARD_DIR = './tensorboard'
config.MODEL_DIR = ''
config.RESULT_DIR = ''
config.DATA_DIR = ''
config.VERBOSE = False
config.TAG = ''
config.DEBUG = False

# CUDNN related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# TAN related params
config.TAN = edict()
config.TAN.OBJECT_MODULE = edict()
config.TAN.OBJECT_MODULE.NAME = ''
config.TAN.OBJECT_MODULE.PARAMS = None
config.TAN.TEXTUAL_MODULE = edict()
config.TAN.TEXTUAL_MODULE.NAME = ''
config.TAN.TEXTUAL_MODULE.PARAMS = None
config.TAN.FRAME_MODULE = edict()
config.TAN.FRAME_MODULE.NAME = ''
config.TAN.FRAME_MODULE.PARAMS = None
config.TAN.PROP_MODULE = edict()
config.TAN.PROP_MODULE.NAME = ''
config.TAN.PROP_MODULE.PARAMS = None
config.TAN.FUSION_MODULE = edict()
config.TAN.FUSION_MODULE.NAME = ''
config.TAN.FUSION_MODULE.PARAMS = None
config.TAN.MAP_MODULE = edict()
config.TAN.MAP_MODULE.NAME = ''
config.TAN.MAP_MODULE.PARAMS = None
config.TAN.PRED_INPUT_SIZE = 512
config.TAN.PRED_MODULE = edict()
config.TAN.PRED_MODULE.NAME = ''
config.TAN.PRED_MODULE.PARAMS = None

# common params for NETWORK
config.MODEL = edict()
config.MODEL.NAME = ''
config.MODEL.CHECKPOINT = '' # The checkpoint for the best performance

# DATASET related params
config.DATASET = edict()
config.DATASET.ROOT = ''
config.DATASET.NAME = ''
config.DATASET.MODALITY = ''
config.DATASET.VIS_INPUT_TYPE = ''
config.DATASET.NO_VAL = False
config.DATASET.BIAS = 0
config.DATASET.NUM_SAMPLE_CLIPS = 256
config.DATASET.TARGET_STRIDE = 16
config.DATASET.DOWNSAMPLING_STRIDE = 16
config.DATASET.SPLIT = ''
config.DATASET.NORMALIZE = False
config.DATASET.RANDOM_SAMPLING = False
config.DATASET.OBJECT_NUM = 18
config.DATASET.NEGATIVE_SAMPLE = False
config.DATASET.PRE_LOAD = False

# train
config.TRAIN = edict()
config.TRAIN.LR = 0.001
config.TRAIN.WEIGHT_DECAY = 0
config.TRAIN.FACTOR = 0.8
config.TRAIN.PATIENCE = 20
config.TRAIN.MAX_EPOCH = 20
config.TRAIN.BATCH_SIZE = 4
config.TRAIN.STEP_ACCUMULATE = 1
config.TRAIN.SHUFFLE = True
config.TRAIN.CONTINUE = False
config.TRAIN.FP16 = False
config.TRAIN.FINE_TUNE = False

config.LOSS = edict()
config.LOSS.NAME = 'bce_loss'
config.LOSS.PARAMS = None

# test
config.TEST = edict()
config.TEST.RECALL = []
config.TEST.TIOU = []
config.TEST.NMS_THRESH = 0.4
config.TEST.INTERVAL = 1
config.TEST.EVAL_TRAIN = False
config.TEST.BATCH_SIZE = 1
config.TEST.TOP_K = 10

def _update_dict(cfg, value):
    for k, v in value.items():
        if k in cfg:
            if k == 'PARAMS':
                cfg[k] = v
            elif isinstance(v, dict):
                _update_dict(cfg[k],v)
            else:
                cfg[k] = v
        else:
            raise ValueError("{} not exist in config.py".format(k))

def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(config[k], v)
                else:
                    config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))
