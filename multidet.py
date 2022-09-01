import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
import math

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
####################

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h

        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(3, d_model)
        self.linear2 = nn.Linear(3, d_model)
        self.linear3 = nn.Linear(3, d_model)

    def forward(self, query, key, value):
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=None,dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
####################

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
####################

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
####################

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))
####################

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
####################

class EncoderLayer(nn.Module):
    def __init__(self, d_model, dropout):
        super(EncoderLayer, self).__init__()
        self.size = d_model
        self.self_attn = MultiHeadedAttention(h=8, d_model=d_model, dropout=0.1)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=128, dropout=0.1)
        self.sublayer = clones(SublayerConnection(size=d_model, dropout=0.1), 2)

    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)
####################

class Encoder(nn.Module):
    def __init__(self, N, d_model):
        super(Encoder, self).__init__()
        layer = EncoderLayer(d_model=d_model,dropout=0.1)
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
####################

class DecoderLayer(nn.Module):
    def __init__(self, d_model, dropout):
        super(DecoderLayer, self).__init__()
        self.size = d_model
        self.self_attn = MultiHeadedAttention(h=8, d_model=d_model, dropout=0.1)
        self.src_attn = MultiHeadedAttention(h=8, d_model=d_model, dropout=0.1)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=128, dropout=0.1)
        self.sublayer = clones(SublayerConnection(size=d_model, dropout=0.1), 3)

    def forward(self, x, memory):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m))
        return self.sublayer[2](x, self.feed_forward)
####################

class Decoder(nn.Module):
    def __init__(self, N, d_model):
        super(Decoder, self).__init__()
        layer = DecoderLayer(d_model=d_model,dropout=0.1)
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.outputLinear = nn.Linear(d_model, 512)
        self.outputLinear1 = nn.Linear(72*512, 72)
        self.outputLinear2 = nn.Linear(72, 24)

    def forward(self, memory, x):
        for layer in self.layers:
            x = layer(x, memory)

        return self.outputLinear2(self.outputLinear1(torch.flatten(input = self.outputLinear(self.norm(x)),start_dim=1)))
####################

class MultiDeT(nn.Module):
    def __init__(self):
        super(MultiDeT, self).__init__()
        d_model = 32
        self.encoder = Encoder(4,d_model=d_model)
        self.decoder1 = Decoder(4,d_model=d_model)
        self.decoder2 = Decoder(4,d_model=d_model)
        self.decoder3 = Decoder(4,d_model=d_model)

        self.encoderLinear = nn.Linear(10, d_model)
        self.decoder1_Linear = nn.Linear(1, d_model)
        self.decoder2_Linear = nn.Linear(1, d_model)
        self.decoder3_Linear = nn.Linear(1, d_model)

    def forward(self, src):

        encoder_output = self.encoder(self.encoderLinear(src))

        input1 = src[ : , : , : 1 ]
        input2 = src[ : , : , 1 : 2]
        input3 = src[ : , : , 2 : 3]
        decoder1_output = self.decoder1(encoder_output, self.decoder1_Linear(input1))
        decoder2_output = self.decoder2(encoder_output, self.decoder2_Linear(input2))
        decoder3_output = self.decoder3(encoder_output, self.decoder3_Linear(input3))

        out1 = decoder1_output*0.98 + decoder2_output*0.01 + decoder3_output*0.01
        out2 = decoder1_output*0.01 + decoder2_output*0.98 + decoder3_output*0.01
        out3 = decoder1_output*0.01 + decoder2_output*0.01 + decoder3_output*0.98

        return decoder1_output,decoder2_output,decoder3_output
####################
