import os
import math
import argparse
import pickle as pkl

from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import _init_paths
from core.engine import Engine
import datasets
import models
from core.utils import AverageMeter
from core.config import config, update_config
from core.eval import eval_predictions, display_results
import models.loss as loss
import torch.distributed as dist

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.autograd.set_detect_anomaly(True)


def parse_args():
    parser = argparse.ArgumentParser(description='Test localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    # testing
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--dataDir', help='data path', type=str)
    parser.add_argument('--modelDir', help='model path', type=str)
    parser.add_argument('--logDir', help='log path', type=str)
    parser.add_argument('--split', default='val', required=True, choices=['train', 'val', 'test'], type=str)
    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    parser.add_argument('--checkpoint', help='checkpoint path', type=str)
    parser.add_argument('--local_rank', help='local rank', type=int, default=0)
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
        config.OUTPUT_DIR = args.modelDir
    if args.logDir:
        config.LOG_DIR = args.logDir
    if args.verbose:
        config.VERBOSE = args.verbose
    if args.checkpoint:
        config.MODEL.CHECKPOINT = args.checkpoint


def synchronize():
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
    dist.barrier()


def save_scores(scores, data, dataset_name, split):
    results = {}
    for i, d in enumerate(data):
        results[d['video']] = scores[i]
    pkl.dump(results, open(os.path.join(config.RESULT_DIR, dataset_name,
                                        '{}_{}_{}.pkl'.format(config.MODEL.NAME, config.DATASET.VIS_INPUT_TYPE, split)),
                           'wb'))


def gather_tensor(data: torch.Tensor, dim=0, dst: int = None):
    N = dist.get_world_size()
    if N == 1:
        return data
    is_dst = dst is None or dst == dist.get_rank()
    ## get tensor size
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
    ## pad to same shape
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


def save_to_txt(scores, data, dataset_name, split):
    txt_path = os.path.join(config.RESULT_DIR, dataset_name,
                            '{}_{}_{}.txt'.format(config.MODEL.NAME, config.DATASET.VIS_INPUT_TYPE, split))
    rootfolder1 = os.path.dirname(txt_path)
    rootfolder2 = os.path.dirname(rootfolder1)
    if not os.path.exists(rootfolder2):
        print('Make directory %s ...' % rootfolder2)
        os.mkdir(rootfolder2)
    if not os.path.exists(rootfolder1):
        print('Make directory %s ...' % rootfolder1)
        os.mkdir(rootfolder1)

    with open(txt_path, "w") as fb:
        for i, d in enumerate(data):
            fb.write('{} {} == {} {} = {} {} {}\n'.format(d['video'], d['description'], d['times'][0], d['times'][1],
                                                          scores[i][0], scores[i][1], scores[i][2]))


def main():
    args = parse_args()
    reset_config(config, args)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

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
    model = getattr(models, model_name)()

    if distribute:
        torch.distributed.init_process_group(backend="nccl")
        synchronize()

    model = model.cuda(args.local_rank)
    if distribute:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    model_checkpoint = torch.load(config.MODEL.CHECKPOINT,
                                  map_location=lambda storage, loc: storage.cuda(args.local_rank))
    if 'model' in model_checkpoint:
        model.load_state_dict(model_checkpoint['model'] )
    else:
        model.module.load_state_dict(model_checkpoint)

    model.eval()

    test_dataset = getattr(datasets, dataset_name)(args.split)

    def get_sampler(i_dataset, shuffle=False):
        if distribute:
            return torch.utils.data.distributed.DistributedSampler(i_dataset, shuffle=shuffle)
        elif shuffle:
            return torch.utils.data.RandomSampler(i_dataset)
        else:
            return torch.utils.data.SequentialSampler(i_dataset)

    sampler = get_sampler(test_dataset, shuffle=False)
    dataloader = DataLoader(test_dataset,
                            batch_size=config.TRAIN.BATCH_SIZE // num_gpus,
                            num_workers=config.WORKERS // num_gpus,
                            pin_memory=True,
                            sampler=sampler,
                            collate_fn=test_dataset.get_collate_fn())

    def network(sample, just_profile=True):
        if model_name == 'LEORN_F':
            visual_input = sample['batch_vis_input'].cuda(non_blocking=True)
        # anno_idxs = sample['batch_anno_idxs']
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

    def on_test_start(state):
        state['loss_meter'] = AverageMeter()
        if main_work:
            state['index_list'] = []
            state['sorted_segments_list'] = []
            if config.VERBOSE:
                state['progress_bar'] = tqdm(total=math.ceil(len(test_dataset) / config.TEST.BATCH_SIZE))

    def on_test_forward(state):
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

            gather_segment = [torch.ones_like(sum_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(gather_segment, sum_tensor)


            if main_work:
                state['index_list'].extend(
                    [i[:, 0, 0].int().cpu() for i in gather_segment])  # batchsize * num_clips+1 * 2
                state['sorted_segments_list'].extend([i[:, 1:, :].cpu() for i in gather_segment])
        else:
            min_idx = min(state['sample']['batch_anno_idxs'])
            batch_indexs = [idx - min_idx for idx in state['sample']['batch_anno_idxs']]
            sorted_segments = [state['output'][i] for i in batch_indexs]
            state['sorted_segments_list'].extend(sorted_segments)

    def on_test_end(state):
        synchronize()
        if main_work:
            index_list = torch.cat(state['index_list'], dim=0)  # all_video
            sorted_index, inv_index = torch.sort(index_list)

            sorted_segment_list = torch.cat(state['sorted_segments_list'], dim=0)  # all_video * num_clips * 2
            final_segment_list = sorted_segment_list[inv_index, :, :]

            annotations = state['iterator'].dataset.annotations

            del_repeat = final_segment_list.size(0) - len(annotations)
            for i in range(del_repeat):
                final_segment_list = final_segment_list[torch.arange(final_segment_list.size(0)) != i]

            state['Rank@N,mIoU@M'], state['miou'] = eval_predictions(final_segment_list.tolist(), annotations,
                                                                     verbose=False)
            if config.VERBOSE:
                state['progress_bar'].close()

            loss_message = '\ntest loss {:.4f}'.format(state['loss_meter'].avg)
            print(loss_message)
            state['loss_meter'].reset()
            test_table = display_results(state['Rank@N,mIoU@M'], state['miou'],
                                         'performance on testing set')
            table_message = '\n' + test_table
            print(table_message)
            save_to_txt(final_segment_list.tolist(), annotations, config.DATASET.NAME, args.split)
            # os._exit()

    engine = Engine()
    engine.hooks['on_test_start'] = on_test_start
    engine.hooks['on_test_forward'] = on_test_forward
    engine.hooks['on_test_end'] = on_test_end
    engine.test(network, dataloader, args.split)


if __name__ == '__main__':
    main()
