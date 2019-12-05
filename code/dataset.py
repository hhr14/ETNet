import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ETDataset(Dataset):
    def __init__(self, input, output, step_per_epoch, hparams):
        self.input = input
        self.output = output
        self.hparams = hparams
        self.batch_size = hparams.batch_size
        self.window_size = hparams.window_size
        self.step_per_epoch = step_per_epoch
        self.content_size = hparams.content_size
        self.refer_size = hparams.refer_size
        self.output_size = hparams.output_size
        self.sample_list = []
        self.build_random_list()

    def __getitem__(self, item):

        random_index = np.random.randint(0, len(self.sample_list))
        id = self.sample_list[random_index][0]
        begin = self.sample_list[random_index][1]
        #  now input is : [ppg-218, mfcc-12+1, others-27, emotion-4]
        content_input = self.input[id][begin: begin + self.window_size, :self.content_size]
        # refer_input = self.input[id][begin: begin + self.window_size,
        #                              self.content_size: self.content_size + self.refer_size]
        refer_input = self.input[id][begin: begin + self.window_size, :self.content_size]
        fwh_output = np.concatenate((self.output[id][begin: begin + self.window_size, 1: 28],
                                     self.output[id][begin: begin + self.window_size, 33: 60],
                                     self.output[id][begin: begin + self.window_size, 65: 92]), axis=1)

        return {'content': content_input, 'refer': refer_input, 'label': fwh_output}

    def __len__(self):
        # len must be real length of dataset, not size / batch_size
        return self.step_per_epoch * self.batch_size

    def build_random_list(self):
        for i in range(len(self.input)):
            if len(self.input[i]) > self.window_size:
                for j in range(len(self.input[i]) - self.window_size):
                    self.sample_list.append([i, j])
        print("sample_list length:", len(self.sample_list))