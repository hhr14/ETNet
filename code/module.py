import torch.nn as nn
import torch
import torch.nn.functional as F


class GSTReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size * 2=256]
    '''

    def __init__(self, hp):

        super().__init__()
        self.hp = hp
        K = len(hp.ref_enc_filters)
        filters = [1] + hp.ref_enc_filters
        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=hp.ref_enc_filters[i]) for i in range(K)])

        out_channels = self.calculate_channels(hp.refer_size, 3, 2, 1, K)
        # self.gru = nn.GRU(input_size=hp.ref_enc_filters[-1] * out_channels,
        #                   hidden_size=hp.ref_gru_size,
        #                   batch_first=True)
        self.gru = nn.GRU(input_size=hp.refer_size,
                          hidden_size=hp.ref_enc_gru_size,
                          batch_first=True,
                          num_layers=hp.ref_enc_gru_layers,
                          bidirectional=True)

    def forward(self, inputs):
        # N = inputs.size(0)
        # out = inputs.view(N, 1, -1, self.hp.refer_size)  # [N, 1, Ty, n_mels]
        # for conv, bn in zip(self.convs, self.bns):
        #     out = conv(out)
        #     out = bn(out)
        #     out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]
        #
        # out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        # T = out.size(1)
        # N = out.size(0)
        # out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]
        #
        # self.gru.flatten_parameters()
        # memory, out = self.gru(out)  # out --- [1, N, E//2]
        #
        # return out.squeeze(0)
        output, hn = self.gru(inputs)
        return output

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class DNNReferenceEncoder(nn.Module):
    def __init__(self, hparams, input_size):
        super().__init__()
        self.hparams = hparams
        size_list = [input_size] + hparams.RefDenseList

        self.RefDense = nn.ModuleList([nn.Sequential(nn.Linear(in_features=size_list[i],
                                                               out_features=size_list[i + 1]),
                                                     nn.Dropout(p=hparams.dprate))
                                       for i in range(len(hparams.RefDenseList))])

    def forward(self, y):
        # input : MFCC/STFT [batch_size, time_step, mel]
        # output : Encoder_seq [batch_size, time_step, RE_size]
        for i, dense in enumerate(self.RefDense):
            y = dense(y)
        return y


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads):

        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units], no request for query_dim and key_dim.
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores_softmax = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores_softmax, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out, scores, scores_softmax


class DotAttention(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

    def forward(self, query, key):
        # query : [b, to, h], key: [b, ti, h]
        a = torch.bmm(query, key.transpose(1,2))  # a,e : [b, to, ti]
        e = nn.Softmax(dim=-1)(a)
        c = torch.bmm(e, key)  # c : [b, to, h]
        return c


#  following part is TacotronContentEncoder


class TacotronContentEncoder(nn.Module):
    '''
    input:
        inputs: [N, T_x, E]
    output:
        outputs: [N, T_x, E]
        hidden: [2, N, E//2]
    '''

    def __init__(self, hp):
        super().__init__()
        self.prenet = PreNet(in_features=hp.refer_size, hp=hp)  # [N, T, E//2]

        self.conv1d_bank = Conv1dBank(K=hp.K, in_channels=hp.E // 2, out_channels=hp.E // 2)  # [N, T, E//2 * K]

        self.conv1d_1 = Conv1d(in_channels=hp.K * hp.E // 2, out_channels=hp.E // 2, kernel_size=3)  # [N, T, E//2]
        self.conv1d_2 = Conv1d(in_channels=hp.E // 2, out_channels=hp.E // 2, kernel_size=3)  # [N, T, E//2]
        self.bn1 = BatchNorm1d(num_features=hp.E // 2)
        self.bn2 = BatchNorm1d(num_features=hp.E // 2)

        self.highways = nn.ModuleList([Highway(in_features=hp.E // 2, out_features=hp.E // 2)
                                       for i in range(hp.num_highways)])

        self.gru = nn.GRU(input_size=hp.E // 2, hidden_size=hp.E // 2, num_layers=2, bidirectional=True, batch_first=True)

    def forward(self, inputs, prev_hidden=None):
        # prenet
        inputs = self.prenet(inputs)  # [N, T, E//2]

        # CBHG
        # conv1d bank
        outputs = self.conv1d_bank(inputs)  # [N, T, E//2 * K]
        outputs = max_pool1d(outputs, kernel_size=2)  # [N, T, E//2 * K]

        # conv1d projections
        outputs = self.conv1d_1(outputs)  # [N, T, E//2]
        outputs = self.bn1(outputs)
        outputs = nn.functional.relu(outputs)  # [N, T, E//2]
        outputs = self.conv1d_2(outputs)  # [N, T, E//2]
        outputs = self.bn2(outputs)

        outputs = outputs + inputs  # residual connect

        # highway
        for i, layer in enumerate(self.highways):
            outputs = layer(outputs)
            # outputs = nn.functional.relu(outputs)  # [N, T, E//2]

        # outputs = torch.transpose(outputs, 0, 1)  # [T, N, E//2]

        self.gru.flatten_parameters()
        outputs, hidden = self.gru(outputs, prev_hidden)  # outputs [N, T, E]

        return outputs, hidden


class PreNet(nn.Module):
    '''
    inputs: [N, T, in]
    outputs: [N, T, E // 2]
    '''

    def __init__(self, in_features, hp):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hp.E)
        self.linear2 = nn.Linear(hp.E, hp.E // 2)
        self.dropout1 = nn.Dropout(hp.dropout_p)
        self.dropout2 = nn.Dropout(hp.dropout_p)

    def forward(self, inputs):
        # print(inputs.data.cpu().numpy())
        outputs = self.linear1(inputs)
        outputs = F.relu(outputs)
        outputs = self.dropout1(outputs)
        outputs = self.linear2(outputs)
        outputs = F.relu(outputs)
        outputs = self.dropout2(outputs)
        return outputs


class Conv1dBank(nn.Module):
    '''
        inputs: [N, T, C_in]
        outputs: [N, T, C_out * K]  # same padding
    Args:
        in_channels: E//2
        out_channels: E//2
    '''

    def __init__(self, K, in_channels, out_channels):
        super().__init__()
        self.bank = nn.ModuleList([Conv1d(in_channels, out_channels, kernel_size=k)
                                   for k in range(1, K + 1)])
        self.bn = BatchNorm1d(out_channels * K)

    def forward(self, inputs):

        output_list = []
        for k, banklayer in enumerate(self.bank):
            output = banklayer(inputs)
            output_list.append(output)
        outputs = torch.cat(output_list, dim=2)
        outputs = self.bn(outputs)  # [N, T, C_out * K]
        outputs = F.relu(outputs)

        return outputs


class BatchNorm1d(nn.Module):
    '''
    inputs: [N, T, C]
    outputs: [N, T, C]
    '''
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, inputs):
        out = self.bn(inputs.transpose(1, 2).contiguous())
        return out.transpose(1, 2)


def max_pool1d(inputs, kernel_size, stride=1, padding='same'):
    '''
    inputs: [N, T, C]
    outputs: [N, T // stride, C]
    '''
    inputs = inputs.transpose(1, 2)  # [N, C, T]
    if padding == 'same':
        left = (kernel_size - 1) // 2
        right = (kernel_size - 1) - left
        pad = [left, right]
    else:
        pad = [0, 0]
    inputs = F.pad(inputs, pad)
    outputs = F.max_pool1d(inputs, kernel_size, stride)  # [N, C, T]
    outputs = outputs.transpose(1, 2)  # [N, T, C]

    return outputs


class Conv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        '''
        inputs: [N, T, C_in]
        outputs: [N, T, C_out]
        '''
        super().__init__()
        self.pad = [0, 0]
        if padding == 'same':
            left = (kernel_size - 1) // 2
            right = (kernel_size - 1) - left
            self.pad = [left, right]
            # pad = kernel_size // 2
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride)

    def forward(self, inputs):
        inputs = torch.transpose(inputs, 1, 2)  # [N, C_in, T]
        inputs = F.pad(inputs, self.pad)
        out = self.conv1d(inputs)  # [N, C_out, T]
        out = torch.transpose(out, 1, 2)  # [N, T, C_out]
        return out


class Highway(nn.Module):

    def __init__(self, in_features, out_features):
        '''
        inputs: [N, T, C]
        outputs: [N, T, C]
        '''
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(in_features, out_features)

    def forward(self, inputs):
        H = self.linear1(inputs)
        H = F.relu(H)
        T = self.linear2(inputs)
        T = F.sigmoid(T)

        out = H * T + inputs * (1.0 - T)

        return out
