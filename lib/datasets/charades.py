""" Dataset loader for the Charades-STA dataset """
import os
import csv

import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext

from . import average_to_fixed_length
from core.eval import iou
from core.config import config
import datasets


class Charades(data.Dataset):
    vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
    vocab.itos.extend(['<unk>'])
    vocab.stoi['<unk>'] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)

    def __init__(self, split, rand_clip=False, training=False):
        super(Charades, self).__init__()

        self.vis_input_type = config.DATASET.VIS_INPUT_TYPE
        self.data_dir = config.DATA_DIR
        self.split = split

        self.durations = {}
        with open(os.path.join(self.data_dir, 'Charades_v1_{}.csv'.format(split))) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.durations[row['id']] = float(row['length'])

        anno_file = open(os.path.join(self.data_dir, "charades_sta_{}.txt".format(self.split)), 'r')
        annotations = []
        for line in anno_file:
            anno, sent = line.split("##")
            sent = sent.split('.\n')[0]
            vid, s_time, e_time = anno.split(" ")
            s_time = float(s_time)
            e_time = min(float(e_time), self.durations[vid])
            if s_time < e_time:
                annotations.append(
                    {'video': vid, 'times': [s_time, e_time], 'description': sent, 'duration': self.durations[vid]})
        anno_file.close()
        self.annotations = annotations
        self.rand_clip = rand_clip

    def __getitem__(self, index):
        video_id = self.annotations[index]['video']
        gt_s_time, gt_e_time = self.annotations[index]['times']
        description = self.annotations[index]['description']
        duration = self.durations[video_id]

        word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in description.split()],
                                 dtype=torch.long)
        word_vectors = self.word_embedding(word_idxs)

        visual_input, visual_mask = self.get_video_features(video_id)

        if self.rand_clip:
            visual_input, duration, gt_s_time, gt_e_time = random_clip(visual_input, duration, gt_s_time, gt_e_time)

        # Time scaled to fixed size
        # visual_input = sample_to_fixed_length(visual_input, random_sampling=True)
        # visual_input = interpolate_to_fixed_length(visual_input)
        visual_input = average_to_fixed_length(visual_input)
        num_clips = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE   # 16
        s_times = torch.arange(0, num_clips).float() * duration / num_clips
        e_times = torch.arange(1, num_clips + 1).float() * duration / num_clips
        overlaps = iou(torch.stack([s_times[:, None].expand(-1, num_clips),
                                    e_times[None, :].expand(num_clips, -1)], dim=2).view(-1, 2).tolist(),
                       torch.tensor([gt_s_time, gt_e_time]).tolist()).reshape(num_clips, num_clips)

        gt_s_idx = np.argmax(overlaps) // num_clips
        gt_e_idx = np.argmax(overlaps) % num_clips

        item = {
            'visual_input': visual_input,
            'anno_idx': index,
            'word_vectors': word_vectors,
            'txt_mask': torch.ones(word_vectors.shape[0], 1),
            'map_gt': torch.from_numpy(overlaps),
            'reg_gt': torch.tensor([gt_s_idx, gt_e_idx]),
            'duration': duration,
            'description': description,
        }

        return item

    def __len__(self):
        return len(self.annotations)

    def get_collate_fn(self):
        return datasets.collate_fn

    def get_video_features(self, vid):
        hdf5_file = h5py.File(os.path.join(self.data_dir, '{}_features.hdf5'.format(self.vis_input_type)), 'r')
        features = torch.from_numpy(hdf5_file[vid][:]).float()
        if config.DATASET.NORMALIZE:
            features = F.normalize(features, dim=1)
        vis_mask = torch.ones((features.shape[0], 1))
        return features, vis_mask
