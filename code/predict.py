from hparams import get_hparams
import os
import torch
import Utils
import pickle
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from skimage import io


def predict(hparams, predict_file_list, ppg_scaler, fwh_scaler, mymodel, device):
    # use model.eval() to ban dropout or BN
    mymodel.eval()
    for i in range(len(predict_file_list)):
        print('predict ' + predict_file_list[i], ' .....')
        input = np.load(predict_file_list[i])
        input = ppg_scaler.transform(input)
        fwh_predict = np.zeros((input.shape[0], hparams.output_size))
        fwh_count = np.zeros((input.shape[0], 1))
        attention_predict = np.zeros((hparams.num_heads, input.shape[0], hparams.tokenNum))
        begin = 0
        end = begin + hparams.window_size
        while end <= input.shape[0]:
            content_input = input[begin: end, : hparams.content_size]
            content_input = torch.from_numpy(content_input).to(device)
            ref_input = input[begin: end, hparams.content_size: hparams.content_size + hparams.refer_size]
            ref_input = ref_input[np.newaxis, :]
            content_input = content_input[np.newaxis, :]
            ref_input = torch.from_numpy(ref_input).to(device)
            output = mymodel(ref_input.float(), content_input.float())
            attention_predict[:, begin: end, :] += mymodel.attention.squeeze()
            fwh_predict[begin: end, :] += (output.cpu().detach().numpy()).reshape((-1, hparams.output_size))
            fwh_count[begin: end, :] += 1
            begin += hparams.predict_step
            end += hparams.predict_step
        fwh_predict /= fwh_count
        attention_predict /= fwh_count
        if hparams.output_size == 81:
            fwh_res = np.ones((input.shape[0], 96))
            fwh_res[:, 1: 28] = fwh_predict[:, 0: 27]
            fwh_res[:, 33: 60] = fwh_predict[:, 27: 54]
            fwh_res[:, 65: 92] = fwh_predict[:, 54: 81]
            fwh_predict = fwh_res
        fwh_predict = fwh_scaler.inverse_transform(fwh_predict)
        input_file_name = (predict_file_list[i].split('/')[-1]).split('.')[0]

        np.save(os.path.join(hparams.output_folder, input_file_name + '_fwh32'), fwh_predict)

        if hparams.plotattn:
            plot_attention(attention_predict, input_file_name)


def plot_attention(attention, file_name):
    if os.path.exists(hparams.attention_path + file_name + '_attn/') is False:
        os.mkdir(hparams.attention_path + file_name + '_attn/')
    for i in range(hparams.num_heads):
        matrix = attention[i].transpose()
        matrix = np.array([[int(255 * matrix[i][j]) for j in range(matrix.shape[1])]
                           for i in range(matrix.shape[0])], dtype='uint8')
        io.imsave(hparams.attention_path + file_name + '_attn/head_' + str(i) + '.jpg', matrix)


if __name__ == "__main__":
    hparams = get_hparams()
    print(hparams)
    os.environ['CUDA_VISIBLE_DEVICES'] = hparams.gpu
    device = torch.device("cuda")
    mymodel = Utils.create_model(hparams, mode='predict')
    mymodel = mymodel.float()
    mymodel.to(device)
    (ppg_scaler, fwh_scaler, shape_info) = pickle.load(open(os.path.join(hparams.dataset,
                                                                         'scaler.pickle'), 'rb'))
    predict(hparams, Utils.get_predict_file_list(hparams.predict), ppg_scaler, fwh_scaler, mymodel, device)
