from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_logger(cfg, cfg_name, tag='train'):
    root_log_dir = Path(cfg.LOG_DIR)
    # set up logger
    if not root_log_dir.exists():
        print('=> creating {}'.format(root_log_dir))
        root_log_dir.mkdir()

    dataset = cfg.DATASET.NAME
    cfg_name = os.path.basename(cfg_name).split('.yaml')[0]

    final_log_dir = root_log_dir / dataset / cfg_name

    print('=> creating {}'.format(final_log_dir))
    final_log_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, tag)
    final_log_file = final_log_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    # tensorboard writer

    root_tb_dir = Path(cfg.TENSORBOARD_DIR)
    # set up logger
    if not root_tb_dir.exists():
        print('=> creating {}'.format(root_tb_dir))
        root_tb_dir.mkdir()
    final_tb_dir = root_tb_dir / dataset / cfg_name / time_str
    print('=> creating {}'.format(final_tb_dir))
    final_tb_dir.mkdir(parents=True, exist_ok=True)

    tb_writer = SummaryWriter(str(final_tb_dir))

    return logger, str(final_log_dir), time_str, tb_writer

def create_tf_writer(cfg, cfg_name, is_feature_writer=True, feature_name='default'):
    root_tb_dir = Path(cfg.TENSORBOARD_DIR)
    dataset = cfg.DATASET.NAME
    cfg_name = os.path.basename(cfg_name).split('.yaml')[0]
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    # set up logger
    if not root_tb_dir.exists():
        print('=> creating {}'.format(root_tb_dir))
        root_tb_dir.mkdir()
    if is_feature_writer:
        final_tb_dir = root_tb_dir / 'features' / dataset / cfg_name / feature_name
    else:
        final_tb_dir = root_tb_dir / dataset / cfg_name / time_str
    print('=> creating {}'.format(final_tb_dir))
    final_tb_dir.mkdir(parents=True, exist_ok=True)

    tb_writer = SummaryWriter(str(final_tb_dir))

    return tb_writer

