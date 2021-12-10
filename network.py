import torch.nn as nn
import math


from pos_encoder import PositionalEncoding

import config


class ClusterFormer(nn.Module):

    def __init__(self):
        super().__init__()
        hp = config.CONFIG.model

        self.d_model = hp['d_model']
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hp['d_model'], nhead=hp['nhead'])
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=hp['enc_layers'])
        self.pos_encoder = PositionalEncoding()
        self.encoder = nn.Embedding(hp['ntoken'], hp['d_model'])

    def forward(self, x):
        x = self.encoder(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        out = self.transformer_encoder(x)
        return out
