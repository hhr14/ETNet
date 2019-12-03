import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from module import MultiHeadAttention, GSTReferenceEncoder, DNNReferenceEncoder, DotAttention


class ETNet(nn.Module):
    def __init__(self, hparams, mode='train'):
        super().__init__()
        self.hparams = hparams
        if hparams.encoder_mode == 'GST':
            self.RefEncoder_ = GSTReferenceEncoder(hparams)
        elif hparams.encoder_mode == 'DNN':
            self.RefEncoder_ = DNNReferenceEncoder(hparams, hparams.refer_size)
        else:
            raise ValueError('Illegal ref_encoder mode accepted!')
        self.ETLayer_ = ETLayer(hparams)
        self.ContentEncoder_ = ContentEncoder(hparams, hparams.content_size)
        self.Decoder_ = Decoder(hparams)
        self.mode = mode
        self.attention = None

    def forward(self, audio_input, txt_input):
        refEmbedding = self.RefEncoder_(audio_input)
        ETEmbedding, attention_ = self.ETLayer_(refEmbedding)
        if self.mode == 'predict':
            self.attention = attention_.cpu().detach().numpy()
        ContentEmbedding = self.ContentEncoder_(txt_input)
        if self.hparams.emotion_add_mode == 'cat':
            decoderInput = torch.cat([ETEmbedding, ContentEmbedding], dim=-1)
        elif self.hparams.emotion_add_mode == 'add':
            decoderInput = torch.add(ETEmbedding, ContentEmbedding)
        else:
            raise ValueError('Illegal add mode accepted!')
        decoderOutput = self.Decoder_(decoderInput)
        return decoderOutput


class ContentEncoder(nn.Module):
    def __init__(self, hparams, input_size):
        super().__init__()
        self.hparams = hparams
        size_list = [input_size] + hparams.ContentDenseList

        self.ContentEncoderDense = nn.ModuleList([nn.Sequential(nn.Linear(in_features=size_list[i],
                                                                          out_features=size_list[i + 1]),
                                                  nn.Dropout(p=hparams.dprate))
                                                  for i in range(len(hparams.ContentDenseList))])

    def forward(self, y):
        for i, dense in enumerate(self.ContentEncoderDense):
            y = dense(y)
        return y


class Decoder(nn.Module):
    def __init__(self, hparams):
        # LSTM input : [seq_len, batch_size, input_size]
        # LSTM output : [seq_len, batch_size, 2 * hidden_size]  <---  default concatenate!!!
        # if batch first then both input & output become [batch_size, seq_len]
        # zoneout!!! activations!!!
        super().__init__()
        self.DecoderGRU = nn.GRU(input_size=hparams.decoderInputSize,
                                 hidden_size=hparams.decoderRNNunits,
                                 num_layers=hparams.decoderRNNnum,
                                 batch_first=True, bidirectional=True)
        size_list = [2 * hparams.decoderRNNunits] + hparams.DecoderDenseList
        self.DecoderDense = nn.ModuleList([nn.Sequential(nn.Linear(in_features=size_list[i],
                                                                   out_features=size_list[i + 1]),
                                                         nn.Dropout(p=hparams.dprate))
                                           for i in range(len(hparams.DecoderDenseList))])

    def forward(self, encoder_states):
        output, hn = self.DecoderGRU(encoder_states)
        for i, dense in enumerate(self.DecoderDense):
            output = dense(output)
        return output


class ETLayer(nn.Module):
    def __init__(self, hparams):
        super(ETLayer, self).__init__()
        self.hparams = hparams
        self.ET = np.random.normal(loc=0.0, scale=1.0, size=(hparams.tokenNum, hparams.tokenSize))
        self.embedding = nn.Parameter(torch.from_numpy(self.ET), requires_grad=True)
        if hparams.attention_mode == 'dot':
            self.attention = DotAttention(hparams)
        elif hparams.attention_mode == 'multihead':
            self.attention = MultiHeadAttention(query_dim=hparams.ref_enc_gru_size * 2,
                                                key_dim=hparams.tokenSize,
                                                num_units=hparams.num_units,
                                                num_heads=hparams.num_heads)
        else:
            raise ValueError('Illegal attention mode accepted!')

    def forward(self, encoder_states):
        batch_size = encoder_states.size(0)
        key = torch.tanh(self.embedding).unsqueeze(0).expand(batch_size, -1, -1)
        ETEmbedding, attention_ = self.attention(encoder_states, key)
        return ETEmbedding, attention_

