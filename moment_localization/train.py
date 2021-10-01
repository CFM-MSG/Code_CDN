from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from datetime import time

import _init_paths
import os
import pprint
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import datasets
import models
from core.config import config, update_config
from core.engine import Engine
from core.utils import AverageMeter
from core import eval
from core.utils import create_logger
import models.loss as loss
import math
import torch.distributed as dist
import time
from torch.autograd import profiler
from prefetch_generator import BackgroundGenerator

torch.manual_seed(0)
torch.cuda.manual_seed(0)


class MTimmer:
    def __init__(self, local_rank):
        self.last_time = time.time()
        self.local_rank = local_rank

    def click(self, message='No massage'):
        if self.local_rank == 2:
            print()
            print('=', self.local_rank, '= ', time.asctime(time.localtime(time.time())), '----',
                  time.time() - self.last_time)
            print(message)
        self.last_time = time.time()


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--dataDir', help='data path', type=str)
    parser.add_argument('--modelDir', help='model path', type=str)
    parser.add_argument('--logDir', help='log path', type=str)
    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    parser.add_argument('--tag', help='tags shown in log', type=str)
    parser.add_argument('--local_rank', help='local rank', type=int, default=0)
    parser.add_argument('--tensorboardDir', help='tensorboard path', type=str)
    parser.add_argument('--debug', default=False, action='store_true', help='enable assert')
    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.dataDir:
        config.DATA_DIR = args.dataDir
    if args.modelDir:
        config.MODEL_DIR = args.modelDir
    if args.logDir:
        config.LOG_DIR = args.logDir
    if args.verbose:
        config.VERBOSE = args.verbose
    if args.tag:
        config.TAG = args.tag
    if args.tensorboardDir:
        config.TENSORBOARD_DIR = args.tensorboardDir
    if args.debug:
        config.DEBUG = args.debug


def synchronize(verbose=False):
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    if verbose:
        print('waiting={}'.format(dist.get_rank()))
    dist.barrier()
    if verbose:
        print('waiting finished={}'.format(dist.get_rank()))


def gather_tensor(data: torch.Tensor, dim=0, dst: int = None):
    N = dist.get_world_size()
    if N == 1:
        return data
    is_dst = dst is None or dst == dist.get_rank()
    # get tensor size
    size = torch.tensor(data.shape[dim], device=data.device)
    size_list = [size.clone() for _ in range(N)] if is_dst else None
    if dst is None:
        torch.distributed.all_gather(size_list, size)
    else:
        torch.distributed.gather(tensor=size, gather_list=size_list, dst=dst)
    max_size = max(size.item() for size in size_list)
    shape = list(data.shape)
    shape[dim] = max_size
    tensor_list = [data.new_empty(shape) for _ in range(N)] if is_dst else None
    # pad to same shape
    if data.shape[dim] != max_size:
        shape[dim] = max_size - data.shape[dim]
        tensor = torch.cat([data, data.new_zeros(shape)], dim=dim)
    else:
        tensor = data
    if dst is None:
        torch.distributed.all_gather(tensor_list, tensor)
    else:
        torch.distributed.gather(tensor, tensor_list, dst)
    if is_dst:
        return torch.cat([x.narrow(dim, 0, n.item()) for n, x in zip(size_list, tensor_list)], dim=dim)
    else:
        return None


def main():
    args = parse_args()
    reset_config(config, args)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    torch.multiprocessing.set_sharing_strategy('file_system')
    if config.DEBUG:
        torch.autograd.set_detect_anomaly(True)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    num_gpus = torch.cuda.device_count()

    if num_gpus > 1:
        distribute = True
    else:
        distribute = False

    if 0 == args.local_rank:
        main_work = True
    else:
        main_work = False

    dataset_name = config.DATASET.NAME
    model_name = config.MODEL.NAME

    torch.cuda.set_device(args.local_rank)

    # logger tensorboard
    if main_work:
        logger, final_output_dir, time_str, tb_writer = create_logger(config, args.cfg, config.TAG)
        logger.info('\n' + pprint.pformat(args))
        logger.info('\n' + pprint.pformat(config))

    train_dataset = getattr(datasets, dataset_name)('train', training=True)

    if config.TEST.EVAL_TRAIN:
        eval_train_dataset = getattr(datasets, dataset_name)('train')
    if not config.DATASET.NO_VAL:
        val_dataset = getattr(datasets, dataset_name)('val')
    test_dataset = getattr(datasets, dataset_name)('test')

    model = getattr(models, model_name)()

    if distribute:
        torch.distributed.init_process_group(backend="nccl")
        synchronize()

    model = model.cuda(args.local_rank)
    if distribute:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    if hasattr(model.module, 'get_parameters'):
        params = model.module.get_parameters()
    else:
        params = model.parameters()

    optimizer = optim.Adam(params, lr=config.TRAIN.LR, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=config.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.TRAIN.FACTOR,
                                                     patience=config.TRAIN.PATIENCE, verbose=config.VERBOSE)

    def iterator(split):
        def get_sampler(i_dataset, shuffle=True):
            if distribute:
                return torch.utils.data.distributed.DistributedSampler(i_dataset, shuffle=shuffle)
            elif shuffle:
                return torch.utils.data.RandomSampler(i_dataset)
            else:
                return torch.utils.data.SequentialSampler(i_dataset)

        if split == 'train':
            sampler = get_sampler(train_dataset)
            dataloader = DataLoader(train_dataset,
                                    batch_size=config.TRAIN.BATCH_SIZE // num_gpus,
                                    num_workers=config.WORKERS // num_gpus,
                                    pin_memory=True,
                                    sampler=sampler,
                                    collate_fn=train_dataset.get_collate_fn())
        elif split == 'val':
            sampler = get_sampler(val_dataset, shuffle=False)
            dataloader = DataLoader(val_dataset,
                                    batch_size=config.TEST.BATCH_SIZE // num_gpus,
                                    num_workers=config.WORKERS // num_gpus,
                                    pin_memory=False,
                                    sampler=sampler,
                                    collate_fn=val_dataset.get_collate_fn())
        elif split == 'test':
            sampler = get_sampler(test_dataset, shuffle=False)
            dataloader = DataLoader(test_dataset,
                                    batch_size=config.TEST.BATCH_SIZE // num_gpus,
                                    num_workers=config.WORKERS // num_gpus,
                                    pin_memory=True,
                                    sampler=sampler,
                                    collate_fn=test_dataset.get_collate_fn())
        elif split == 'train_no_shuffle':
            sampler = get_sampler(eval_train_dataset, shuffle=False)
            dataloader = DataLoader(eval_train_dataset,
                                    batch_size=config.TEST.BATCH_SIZE // num_gpus,
                                    num_workers=config.WORKERS // num_gpus,
                                    pin_memory=True,
                                    sampler=sampler,
                                    collate_fn=eval_train_dataset.get_collate_fn())
        else:
            raise NotImplementedError

        return dataloader

    def network(sample, epoch=0):
        if model_name == 'LEORN_F':
            visual_input = sample['batch_vis_input'].cuda(non_blocking=True)
        textual_input = sample['batch_word_vectors'].cuda(non_blocking=True)
        textual_mask = sample['batch_txt_mask'].cuda(non_blocking=True)
        rcnn_input = sample['batch_rcnn_input'].cuda(non_blocking=True)
        rcnn_mask = sample['batch_rcnn_mask'].cuda(non_blocking=True)
        rcnn_bbox = sample['batch_rcnn_bbox'].cuda(non_blocking=True)
        map_gt = sample['batch_map_gt'].cuda(non_blocking=True)
        duration = sample['batch_duration']

        if model_name == 'LEORN':
            prediction, map_mask = model(textual_input, textual_mask, rcnn_input, rcnn_mask, rcnn_bbox)
        else:
            prediction, map_mask = model(textual_input, textual_mask, visual_input, rcnn_input, rcnn_mask, rcnn_bbox)

        loss_value, joint_prob = getattr(loss, config.LOSS.NAME)(prediction, map_mask, map_gt, config.LOSS.PARAMS)

        if model.training:
            return loss_value, None
        else:
            sorted_times = get_proposal_results(joint_prob, duration)
            return loss_value, torch.stack(sorted_times)  # batchsize * num_clips * 2

    def get_proposal_results(scores, durations):
        # batchsize * 1 * 16 * 16 , batchsize
        # assume all valid scores are larger than one
        out_sorted_times = []
        for score, duration in zip(scores, durations):
            T = score.shape[-1]
            sorted_indexs = np.dstack(
                np.unravel_index(np.argsort(score.cpu().detach().numpy().ravel())[::-1], (T, T))).tolist()
            sorted_indexs = np.array([item for item in sorted_indexs[0] if item[0] <= item[1]]).astype(float)

            sorted_indexs[:, 1] = sorted_indexs[:, 1] + 1
            sorted_indexs = torch.from_numpy(sorted_indexs).cuda()
            target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
            out_sorted_times.append((sorted_indexs.float() / target_size * duration))

        return out_sorted_times

    def get_reg_proposal_results(scores, durations, reg_map):
        # batchsize * 1 * 16 * 16 , batchsize
        # reg_map : batchsize * 2 * 16 * 16
        # assume all valid scores are larger than one
        out_sorted_times = []
        for score, duration, reg in zip(scores, durations, reg_map):
            T = score.shape[-1]
            sorted_index = np.dstack(
                np.unravel_index(np.argsort(score.cpu().detach().numpy().ravel())[::-1], (T, T))).tolist()
            sorted_indexs = np.array([item for item in sorted_index[0] if item[0] <= item[1]]).astype(float)
            sorted_reg = torch.stack([reg[:, s, e] for s, e in sorted_index[0] if s <= e], dim=0)

            sorted_indexs[:, 1] = sorted_indexs[:, 1] + 1
            sorted_indexs = torch.from_numpy(sorted_indexs).cuda()
            target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
            sorted_time = (sorted_indexs.float() / target_size + sorted_reg)
            sorted_time[:, 0] = sorted_time[:, 0].masked_fill(sorted_time[:, 0] < 0, 0)
            sorted_time[:, 1] = sorted_time[:, 1].masked_fill(sorted_time[:, 1] > 1, 1)
            sorted_time = sorted_time * duration
            out_sorted_times.append(sorted_time)

        return out_sorted_times

    def on_start(state):
        state['test_interval'] = math.ceil(len(train_dataset) / config.TRAIN.BATCH_SIZE * config.TEST.INTERVAL)

        if config.TRAIN.FP16:
            state['scaler'] = torch.cuda.amp.GradScaler()

        if config.TRAIN.FINE_TUNE and not config.TRAIN.CONTINUE:
            loc = 'cuda:{}'.format(args.local_rank)
            checkpoint = torch.load(config.MODEL.CHECKPOINT, map_location=loc)
            model.module.load_object_params(checkpoint['model'])

        if config.MODEL.CHECKPOINT and config.TRAIN.CONTINUE:
            loc = 'cuda:{}'.format(args.local_rank)
            checkpoint = torch.load(config.MODEL.CHECKPOINT, map_location=loc)
            if 'optimizer' in checkpoint:
                model.load_state_dict(checkpoint['model'])
                state['optimizer'].load_state_dict(checkpoint['optimizer'])
                state['scheduler'].load_state_dict(checkpoint['scheduler'])
                state['t'] = checkpoint['step'] + 1
                if ('scaler' in checkpoint) and (state['scaler'] is not None) and (checkpoint['scaler'] is not None):
                    state['scaler'].load_state_dict(checkpoint['scaler'])
                state['epoch'] = state['t'] // state['test_interval']
            else:
                if distribute:
                    model.module.load_state_dict(checkpoint)
                else:
                    model.load_state_dict(checkpoint)

        state['loss_meter'] = AverageMeter()

        tious = [float(i) for i in config.TEST.TIOU.split(',')] if isinstance(config.TEST.TIOU, str) else [
            config.TEST.TIOU]
        recalls = [int(i) for i in config.TEST.RECALL.split(',')] if isinstance(config.TEST.RECALL, str) else [
            config.TEST.RECALL]
        state['best'] = [[0 for _ in recalls] for _ in tious]
        state['best_miou'] = 0

        model.train()
        if config.VERBOSE and main_work:
            state['progress_bar'] = tqdm(total=state['test_interval'])

    def on_start_epoch(state):
        if distribute:
            state['iterator'].sampler.set_epoch(state['epoch'])

    def on_forward(state):
        if state['t'] % state['step_accumulate'] == 0 or state['t'] % state['test_interval'] == 0:
            if state['scaler'] is not None:
                state['scaler'].unscale_(state['optimizer'])
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        if distribute:
            dist.all_reduce(state['loss'], op=dist.ReduceOp.SUM)
        state['loss_meter'].update(state['loss'].item() / num_gpus, 1)
        # update the lr of transformer
        if hasattr(model, 'adjust_lr'):
            model.adjust_lr(state['optimizer'], state['t'])

    def on_update(state):  # Save All
        # state['scheduler'].step()
        if config.VERBOSE and main_work:
            state['progress_bar'].update(1)

        if state['t'] % state['test_interval'] == 0:
            model.eval()
            if distribute:
                synchronize()

            if config.VERBOSE and main_work:
                state['progress_bar'].close()

                loss_message = '\nepoch: {} iter: {} train loss {:.4f}'.format(state['epoch'], state['t'],
                                                                               state['loss_meter'].avg)
                tb_writer.add_scalars("LOSS", {'train': state['loss_meter'].avg}, state['t'])

                table_message = ''

            if config.TEST.EVAL_TRAIN and state['t'] % (state['test_interval'] * 2) == 0:
                if distribute:
                    synchronize()
                train_state = engine.test(network, iterator('train_no_shuffle'), 'train')
                if main_work:
                    train_table = eval.display_results(train_state['Rank@N,mIoU@M'], train_state['miou'],
                                                       'performance on training set')
                    eval.write2tensorboard(tb_writer, train_state['Rank@N,mIoU@M'], train_state['miou'], state['t'],
                                           'train')
                    table_message += '\n' + train_table

            if not config.DATASET.NO_VAL:
                if distribute:
                    synchronize()
                val_state = engine.test(network, iterator('val'), 'val')
                state['scheduler'].step(-val_state['loss_meter'].avg)
                if main_work:
                    loss_message += ' val loss {:.4f}'.format(val_state['loss_meter'].avg)
                    tb_writer.add_scalars("LOSS", {'val': val_state['loss_meter'].avg}, state['t'])
                    val_state['loss_meter'].reset()
                    val_table = eval.display_results(val_state['Rank@N,mIoU@M'], val_state['miou'],
                                                     'performance on validation set')
                    eval.write2tensorboard(tb_writer, val_state['Rank@N,mIoU@M'], val_state['miou'], state['t'], 'val')
                    table_message += '\n' + val_table

            if distribute:
                synchronize()
            test_state = engine.test(network, iterator('test'), 'test')
            if main_work:
                loss_message += ' test loss {:.4f}'.format(test_state['loss_meter'].avg)
                tb_writer.add_scalars('LOSS', {'test': test_state['loss_meter'].avg}, state['t'])
                test_state['loss_meter'].reset()
                test_table = eval.display_results(test_state['Rank@N,mIoU@M'], test_state['miou'],
                                                  'performance on testing set')
                eval.write2tensorboard(tb_writer, test_state['Rank@N,mIoU@M'], test_state['miou'], state['t'], 'test')
                table_message += '\n' + test_table

                message = loss_message + table_message + '\n'
                logger.info(message)
                tb_writer.flush()

                # assert if better result
                save_checkpoint = False
                if test_state['miou'] > state['best_miou']:
                    state['best_miou'] = test_state['miou']
                    save_checkpoint = True
                for i_tiou in range(len(state['best'])):
                    for i_recall in range(len(state['best'][i_tiou])):
                        if state['best'][i_tiou][i_recall] < test_state['Rank@N,mIoU@M'][i_tiou][i_recall]:
                            state['best'][i_tiou][i_recall] = test_state['Rank@N,mIoU@M'][i_tiou][i_recall]
                            save_checkpoint = True

                if save_checkpoint:
                    saved_model_filename = os.path.join(config.MODEL_DIR,
                                                        '{}/{}/{}/iter{:06d}-{:.4f}-{:.4f}.pkl'.format(
                                                            dataset_name,
                                                            model_name + '_' + config.DATASET.VIS_INPUT_TYPE, time_str,
                                                            state['t'], test_state['Rank@N,mIoU@M'][0, 0],
                                                            test_state['Rank@N,mIoU@M'][0, 1]))

                    rootfolder1 = os.path.dirname(saved_model_filename)
                    rootfolder2 = os.path.dirname(rootfolder1)
                    rootfolder3 = os.path.dirname(rootfolder2)
                    if not os.path.exists(rootfolder3):
                        print('Make directory %s ...' % rootfolder3)
                        os.mkdir(rootfolder3)
                    if not os.path.exists(rootfolder2):
                        print('Make directory %s ...' % rootfolder2)
                        os.mkdir(rootfolder2)
                    if not os.path.exists(rootfolder1):
                        print('Make directory %s ...' % rootfolder1)
                        os.mkdir(rootfolder1)

                    save_state = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'scalar': state['scaler'].state_dict() if state['scaler'] is not None else None,
                        'step': state['t']
                    }
                    torch.save(save_state, saved_model_filename)

            if config.VERBOSE and main_work:
                state['progress_bar'] = tqdm(total=state['test_interval'])
                state['loss_meter'].reset()
            if distribute:
                synchronize()
            model.train()

    def on_end(state):
        if config.VERBOSE and main_work:
            state['progress_bar'].close()
            tb_writer.close()
        if distribute:
            synchronize()

    def on_test_start(state):
        # model.eval()
        state['loss_meter'] = AverageMeter()
        if main_work:
            state['index_list'] = []
            state['sorted_segments_list'] = []
        if config.VERBOSE and main_work:
            if state['split'] == 'train':
                state['progress_bar'] = tqdm(total=math.ceil(len(eval_train_dataset) / config.TEST.BATCH_SIZE))
            elif state['split'] == 'val':
                state['progress_bar'] = tqdm(total=math.ceil(len(val_dataset) / config.TEST.BATCH_SIZE))
            elif state['split'] == 'test':
                state['progress_bar'] = tqdm(total=math.ceil(len(test_dataset) / config.TEST.BATCH_SIZE))
            else:
                raise NotImplementedError
        # timer.click('start load data')

    def on_test_sample(state):
        # timer.click('load data done')
        pass

    def on_test_forward(state):
        # timer.click('on test forward start')
        if config.VERBOSE and main_work:
            state['progress_bar'].update(1)
        if distribute:
            dist.all_reduce(state['loss'], op=dist.ReduceOp.SUM)
        state['loss_meter'].update(state['loss'].item() / num_gpus, 1)

        if distribute:
            batch_indexs = torch.tensor(state['sample']['batch_anno_idxs']).cuda()  # batchsize
            sorted_segments = state['output']  # batchsize * num_clips * 2
            batch_indexs = batch_indexs[:, None, None].repeat(1, 1, 2)  # batchsize * 1 * 2
            sum_tensor = torch.cat([batch_indexs, sorted_segments], dim=1)
            gather_segment = gather_tensor(sum_tensor)
            if main_work:
                state['index_list'].append(gather_segment[:, 0, 0].int().cpu())  # batchsize * num_clips+1 * 2
                state['sorted_segments_list'].append(gather_segment[:, 1:, :].cpu())
        else:
            min_idx = min(state['sample']['batch_anno_idxs'])
            batch_indexs = [idx - min_idx for idx in state['sample']['batch_anno_idxs']]
            sorted_segments = [state['output'][i] for i in batch_indexs]
            state['sorted_segments_list'].extend(sorted_segments)

        # timer.click('on test forward done. load data')

    def on_test_end(state):
        if distribute:
            synchronize()
        if main_work:
            annotations = state['iterator'].dataset.annotations
            if distribute:
                index_list = torch.cat(state['index_list'], dim=0)  # all_video
                sorted_index, inv_index = torch.sort(index_list)

                sorted_segment_list = torch.cat(state['sorted_segments_list'], dim=0)  # all_video * num_clips * 2
                final_segment_list = sorted_segment_list[inv_index, :, :]

                del_repeat = final_segment_list.size(0) - len(annotations)
                for i in range(del_repeat):
                    final_segment_list = final_segment_list[torch.arange(final_segment_list.size(0)) != i]

                state['Rank@N,mIoU@M'], state['miou'] = eval.eval_predictions(
                    final_segment_list.tolist(), annotations, verbose=False)
            else:
                state['Rank@N,mIoU@M'], state['miou'] = eval.eval_predictions(state['sorted_segments_list'],
                                                                              annotations,
                                                                              verbose=False)
            if config.VERBOSE:
                state['progress_bar'].close()

    engine = Engine()
    engine.hooks['on_start'] = on_start
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_update'] = on_update
    engine.hooks['on_end'] = on_end
    engine.hooks['on_test_start'] = on_test_start
    # engine.hooks['on_test_sample'] = on_test_sample
    engine.hooks['on_test_forward'] = on_test_forward
    engine.hooks['on_test_end'] = on_test_end
    engine.train(network,
                 iterator('train'),
                 maxepoch=config.TRAIN.MAX_EPOCH,
                 optimizer=optimizer,
                 scheduler=scheduler,
                 step_accumulate=config.TRAIN.STEP_ACCUMULATE)
    # engine.test(network, iterator('test'), 'test')
    if main_work:
        os._exit(0)


if __name__ == '__main__':
    main()
