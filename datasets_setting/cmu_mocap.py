from torch.utils.data import Dataset
import numpy as np
from h5py import File
import scipy.io as sio
from utils import data_utils
from matplotlib import pyplot as plt
import torch
import os


class Datasets(Dataset):
    def __init__(self, config, split=0,data_aug=False):
        self.path_to_data = "./datasets/cmu_mocap"
        self.data_aug = data_aug
        # self.in_n = opt.input_n  # 50
        # self.out_n = opt.output_n  # 25
        self.split = split
        self.in_n = config.motion.input_length
        self.out_n= config.motion.target_length
        # if self.split <= 1:
        #     self.out_n = opt.output_n
        # else:
        #     self.out_n = opt.test_output_n
        self.dim_used = np.array(
            [9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 42, 43, 44,
             45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68, 69, 70, 71, 75, 76, 77, 78, 79, 80, 84, 85,
             86, 90, 91, 92, 93, 94, 95, 96, 97, 98, 102, 103, 104, 105, 106, 107, 111, 112, 113])
        self.sample_rate = 2
        self.skip_rate=1
        self.test_manner = "all"

        # acts = data_utils.define_actions_cmu(actions)
        # if actions is None:
        #     acts = ["basketball", "basketball_signal", "directing_traffic", "jumping", "running", "soccer",
        #                "walking","washwindow"]
        # else:
        #     acts = actions


        acts = ["basketball", "basketball_signal", "directing_traffic", "jumping", "running", "soccer",
                   "walking","washwindow"]


        # 训练集
        if split == 0:
            self.path_to_data = os.path.join(self.path_to_data, 'train')
            is_test = False
        # 随机取 8 帧
        elif split == 1:
            self.path_to_data = os.path.join(self.path_to_data, 'test')
            is_test = True
        # 取全部的测试集，制造一个整理的测试集
        elif split == 2:
            self.path_to_data = os.path.join(self.path_to_data, 'test')
            is_test = False


        seq_len = self.in_n + self.out_n
        # nactions = len(actions)
        sampled_seq = []
        complete_seq = []
        key = 0
        self.cmu_data = {}
        self.data_idx = []
        for action_idx in np.arange(len(acts)):
            # action = actions[action_idx]
            action = acts[action_idx]
            path = '{}/{}'.format(self.path_to_data, action)
            count = 0
            for _ in os.listdir(path):
                count = count + 1
            for examp_index in np.arange(count):
                filename = '{}/{}/{}_{}.txt'.format(self.path_to_data, action, action, examp_index + 1)
                print("Reading action {}, subaction {}".format(action, examp_index + 1))
                action_sequence = data_utils.readCSVasFloat(filename)
                n, d = action_sequence.shape
                even_list = range(0, n, self.sample_rate)
                num_frames = len(even_list)
                the_sequence = np.array(action_sequence[even_list, :])
                exptmps = torch.from_numpy(the_sequence).float().cuda()

                xyz = data_utils.expmap2xyz_torch_cmu(exptmps)

                # label_index = acts.index(action)
                # self.cmu_data[key] = (label_index, xyz.view(num_frames, -1).cpu().data.numpy())
                self.cmu_data[key] = xyz.view(num_frames, -1).cpu().data.numpy()
                # self.p3d[key] = p3d.view(num_frames, -1).cpu().data.numpy()
                valid_frames = np.arange(0, num_frames - seq_len + 1, self.skip_rate)
                tmp_data_idx_1 = [key] * len(valid_frames)
                tmp_data_idx_2 = list(valid_frames)
                self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                key += 1

                # xyz = xyz.view(-1, 38 * 3)
                # xyz = xyz.cpu().data.numpy()
                # action_sequence = xyz

        #         even_list = range(0, n, self.sample_rate)
        #         the_sequence = np.array(action_sequence[even_list, :])  # x, 114
        #         num_frames = len(the_sequence)
        #         # 训练集，整体测试集
        #         if (not is_test) or (is_test and self.test_manner == "all"):
        #             fs = np.arange(0, num_frames - seq_len + 1)
        #             fs_sel = fs
        #             for i in np.arange(seq_len - 1):
        #                 fs_sel = np.vstack((fs_sel, fs + i + 1))
        #             fs_sel = fs_sel.transpose()
        #             seq_sel = the_sequence[fs_sel, :]
        #             if len(sampled_seq) == 0:
        #                 sampled_seq = seq_sel
        #                 complete_seq = the_sequence
        #             else:
        #                 sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
        #                 complete_seq = np.append(complete_seq, the_sequence, axis=0)
        #
        #         # 测试集 随机挑选 8
        #         elif self.test_manner == "8":
        #             # 水滴测试
        #             source_seq_len = 50
        #             target_seq_len = 25
        #             total_frames = source_seq_len + target_seq_len
        #             batch_size = 8
        #             SEED = 1234567890
        #             rng = np.random.RandomState(SEED)
        #             for _ in range(batch_size):
        #                 idx = rng.randint(0, num_frames - total_frames)
        #                 seq_sel = the_sequence[idx + (source_seq_len - self.in_n):(idx + source_seq_len + self.out_n),
        #                           :]  # 35， 114
        #                 seq_sel = np.expand_dims(seq_sel, axis=0)
        #                 if len(sampled_seq) == 0:
        #                     sampled_seq = seq_sel
        #                     complete_seq = the_sequence
        #                 else:
        #                     sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
        #                     complete_seq = np.append(complete_seq, the_sequence, axis=0)
        #
        # if not is_test:
        #     data_std = np.std(complete_seq, axis=0)
        #     data_mean = np.mean(complete_seq, axis=0)

        joint_to_ignore = np.array([0, 1, 2, 7, 8, 13, 16, 20, 29, 24, 27, 33, 36])
        dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        self.dimensions_to_use = np.setdiff1d(np.arange(114), dimensions_to_ignore)

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        motion=self.cmu_data[key][fs].astype(np.float32)
        motion=motion[:,self.dim_used]
        # 随机反转 一半一半的几率
        if self.data_aug:
            if torch.rand(1)[0] > .5:
                idx = [i for i in range(motion.shape[0] - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                motion = motion[idx]

        human_motion_input = motion[:self.in_n] / 1000  # meter
        human_motion_target = motion[self.in_n:] / 1000  # meter

        # human_motion_input = human_motion_input.float()
        # human_motion_target = human_motion_target.float()

        return human_motion_input, human_motion_target

