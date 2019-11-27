import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from hparams import get_hparams
import os
import Utils as Utils
from dataset import ETDataset
import torch.optim as optim
import numpy as np
from torchsummary import summary


def train(input_train, input_validate, fwh_train, fwh_validate, mymodel, device, recent_epoch):
    # use model.eval() when validate.
    model_name = hparams.model_save_path + 'b' + str(hparams.batch_size) + \
                 '_lr' + str(hparams.lr) + '_'
    train_generator = ETDataset(input_train, fwh_train, hparams.train_step, hparams)
    train_loader = DataLoader(dataset=train_generator, batch_size=hparams.batch_size,
                              shuffle=True, num_workers=4)
    validate_generator = ETDataset(input_validate, fwh_validate, hparams.validate_step, hparams)
    validate_loader = DataLoader(dataset=validate_generator, batch_size=hparams.batch_size,
                                 shuffle=False, num_workers=4)
    optimizer = optim.Adam(mymodel.parameters(), lr=hparams.lr)
    loss_func = nn.MSELoss()
    for e in range(hparams.epochs):
        print('\n\nepoch:', e)
        epoch_ave_loss_list = {'train': [], 'eval': []}

        for mode in ['train', 'eval']:
            if mode == 'train':
                mymodel.train()
                loader = train_loader
            else:
                mymodel.eval()
                loader = validate_loader
            for b, data in enumerate(loader):
                # content = data['content'].double().to(device)
                # print('content.dtype', content.dtype)
                content = data['content'].to(device)
                refer = data['refer'].to(device)
                label = data['label'].to(device)
                optimizer.zero_grad()
                output = mymodel(refer.float(), content.float())
                loss = loss_func(label.float(), output)
                if mode == 'train':
                    loss.backward()
                    optimizer.step()
                    print('step:', b, 'loss:', loss.item())
                epoch_ave_loss_list[mode].append(loss.item())

            epoch_ave_loss = np.mean(epoch_ave_loss_list[mode])
            print(mode + '_loss:', epoch_ave_loss)

        if (e + 1) % hparams.check_point_distance == 0:
            epoch = e + 1 + recent_epoch
            mymodel_name = model_name + 'e' + str(epoch) + '-' + \
                           format(np.mean(epoch_ave_loss_list['eval']), '.4f') + '.pt'
            torch.save({'model_state': mymodel.state_dict()},
                       mymodel_name)


def evaluate():
    # use model.eval() to ban dropout or BN
    pass


if __name__ == "__main__":
    hparams = get_hparams()
    print(hparams)
    os.environ['CUDA_VISIBLE_DEVICES'] = hparams.gpu
    device = torch.device("cuda")
    mymodel, recent_epoch = Utils.create_model(hparams)
    mymodel = mymodel.float()
    mymodel.to(device)
    print(mymodel)
    summary(mymodel, [(hparams.window_size, hparams.refer_size),
                      (hparams.window_size, hparams.content_size)], batch_size=1, device='cuda')
    ppg_train, ppg_validate, ppg_evaluate, fwh_train, fwh_validate, fwh_evaluate, train_mask, \
    validate_mask, evaluate_mask = Utils.load_data(hparams)
    train(ppg_train, ppg_validate, fwh_train, fwh_validate, mymodel, device, recent_epoch)


