import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model):
        super(Encoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.encoder(x)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model):
        super(Decoder, self).__init__()
        encoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8)
        self.decoder = nn.TransformerDecoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dim2Layer1 = nn.Linear(d_model, 512)
        self.dim2Layer2 = nn.Linear(512, 1)
        self.dim1Layer1 = nn.Linear(72, 256)
        self.dim1Layer2 = nn.Linear(256, 128)
        self.dim1Layer3 = nn.Linear(128, 1)

    def forward(self, memory, x):
        x = self.decoder(x, memory)
        x = self.norm(x)
        # x (batch, 72, 32)
        x = self.dim2Layer1(x)
        x = self.dim2Layer2(x)
        # x (batch, 72, 1)
        x = x.reshape((x.shape[0], x.shape[1]))
        # x (batch, 72)
        x = self.dim1Layer1(x)
        x = self.dim1Layer2(x)
        x = self.dim1Layer3(x)
        return x


class MultiDeT(nn.Module):
    def __init__(self):
        super(MultiDeT, self).__init__()
        d_model = 64
        self.encoder = Encoder(2, d_model=d_model)
        self.decoder1 = Decoder(1, d_model=d_model)
        self.decoder2 = Decoder(1, d_model=d_model)
        self.decoder3 = Decoder(1, d_model=d_model)

        self.encoderLinear = nn.Linear(13, d_model)
        self.decoder1_Linear = nn.Linear(1, d_model)
        self.decoder2_Linear = nn.Linear(1, d_model)
        self.decoder3_Linear = nn.Linear(1, d_model)

    def forward(self, src):
        encoder_output = self.encoder(self.encoderLinear(src))

        input1 = src[:, :, : 1]
        input2 = src[:, :, 1: 2]
        input3 = src[:, :, 2: 3]
        decoder1_output = self.decoder1(encoder_output, self.decoder1_Linear(input1))
        decoder2_output = self.decoder2(encoder_output, self.decoder2_Linear(input2))
        decoder3_output = self.decoder3(encoder_output, self.decoder3_Linear(input3))

        return decoder1_output, decoder2_output, decoder3_output
