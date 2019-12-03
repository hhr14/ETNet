import numpy as np
import os
from model import ETNet
import torch
import torch.nn as nn


def load_data(hparams):
    return np.load(os.path.join(hparams.dataset, 'ppg_train.npy')),\
           np.load(os.path.join(hparams.dataset, 'ppg_validation.npy')),\
           np.load(os.path.join(hparams.dataset, 'ppg_evaluation.npy')),\
           np.load(os.path.join(hparams.dataset, 'fwh_train.npy')),\
           np.load(os.path.join(hparams.dataset, 'fwh_validation.npy')),\
           np.load(os.path.join(hparams.dataset, 'fwh_evaluation.npy')),\
           np.load(os.path.join(hparams.dataset, 'data_mask_train.npy')),\
           np.load(os.path.join(hparams.dataset, 'data_mask_validation.npy')),\
           np.load(os.path.join(hparams.dataset, 'data_mask_evaluation.npy'))


def load_recent_model(model_path):
    """
    返回最近的权重文件
    :param path:
    :return:
    """
    model_list = os.listdir(model_path)
    recent_epoch = -1
    recent_file = None
    for model_file in model_list:
        epoch = int(((model_file.split('_')[-1]).split('-')[0])[1:])
        if epoch > recent_epoch:
            recent_epoch = epoch
            recent_file = model_file
    print('recent file', recent_file)
    if recent_file is None:
        return None
    else:
        return os.path.join(model_path, recent_file), recent_epoch


def load_best_model(model_path, hparams):
    """
    返回loss最小的权重文件
    :param path:
    :return:
    """
    model_list = os.listdir(model_path)
    best_loss = 1e10
    best_file = None
    for model_file in model_list:
        model_loss = float((model_file.split('-')[-1])[:-5])
        if hparams.load_epoch is not None:
            epoch = int(((model_file.split('_')[-1]).split('-')[0])[1:])
            if epoch != hparams.load_epoch:
                continue
        if model_loss < best_loss:
            best_loss = model_loss
            best_file = model_file
    print('best_file', best_file)
    return os.path.join(model_path, best_file)


def create_model(hparams, mode='train'):
    if mode == 'train':
        if load_recent_model(hparams.model_save_path) is not None:
            model_path, recent_epoch = load_recent_model(hparams.model_save_path)
            mymodel = ETNet(hparams)
            checkpoint = torch.load(model_path)
            mymodel.load_state_dict(checkpoint['model_state'])
            return mymodel, recent_epoch
        else:
            return ETNet(hparams), 0
    else:
        mymodel = ETNet(hparams, mode=mode)
        best_file_path = load_best_model(hparams.model_save_path, hparams)
        checkpoint = torch.load(best_file_path)
        mymodel.load_state_dict(checkpoint['model_state'])
        return mymodel


def get_predict_file_list(predict_path):
    if os.path.isfile(predict_path):
        return [predict_path]
    else:
        result = []
        predict_folder = os.listdir(predict_path)
        predict_folder.sort()
        for predict_file in predict_folder:
            result.append(os.path.join(predict_path, predict_file))
        return result
