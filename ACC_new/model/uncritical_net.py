import torch
import torch.nn as nn
from .bert import Embeddings, Block, Config, gelu
import math
import torch.nn.functional as F

class PositionalMapping(nn.Module):
    """
    Positional mapping Layer.
    This layer map continuous input coordinates into a higher dimensional space
    and enable the prediction to more easily approximate a higher frequency function.
    See NERF paper for more details (https://arxiv.org/pdf/2003.08934.pdf)
    """

    def __init__(self, input_dim, L=5, pos_scale=0.01, heading_scale=1.0):
        super(PositionalMapping, self).__init__()
        self.L = L
        self.input_dim = input_dim
        self.output_dim = input_dim * (L*2 + 1)
        # self.scale = scale
        self.pos_scale = pos_scale
        self.heading_scale = heading_scale

    def forward(self, x):

        if self.L == 0:
            return x

        x_scale = x.clone()
        x_scale[:, :, :int(2 * self.input_dim / 3)] = x_scale[:, :, :int(2 * self.input_dim / 3)] * self.pos_scale
        x_scale[:, :, int(2 * self.input_dim / 3):] = x_scale[:, :, int(2 * self.input_dim / 3):] * self.heading_scale

        h = [x]
        PI = 3.1415927410125732
        for i in range(self.L):
            x_sin = torch.sin(2**i * PI * x_scale)
            x_cos = torch.cos(2**i * PI * x_scale)

            x_sin[:, :, :int(2 * self.input_dim / 3)], x_sin[:, :, int(2 * self.input_dim / 3):] = x_sin[:, :, :int(2 * self.input_dim / 3)] / self.pos_scale, x_sin[:, :, int(2 * self.input_dim / 3):] / self.heading_scale
            x_cos[:, :, :int(2 * self.input_dim / 3)], x_cos[:, :, int(2 * self.input_dim / 3):] = x_cos[:, :, :int(2 * self.input_dim / 3)] / self.pos_scale, x_cos[:, :, int(2 * self.input_dim / 3):] / self.heading_scale

            h.append(x_sin)
            h.append(x_cos)

        return torch.cat(h, dim=-1)

class Uncritical_Net(nn.Module):
    def __init__(self, input_dim_V, input_dim_C, output_dim, m_tokens_in):
        super().__init__()
        self.input_dim_V = input_dim_V
        self.input_dim_C = input_dim_C
        self.output_dim = output_dim
        self.m_tokens_in = m_tokens_in
        self.h_dim = 256
        self.C_encoding_dim = 20
        self.set_encoding_dim = 32

        self.M_V = PositionalMapping(input_dim=self.input_dim_V, L=4, pos_scale=0.01, heading_scale=0.01)
        self.M_C = PositionalMapping(input_dim=self.input_dim_C, L=4, pos_scale=0.01, heading_scale=0.01)
    
        self.bert_cfg = Config() 
        self.bert_cfg.dim = self.h_dim
        self.bert_cfg.n_layers = 4
        self.bert_cfg.n_heads = 4
        self.bert_cfg.max_len = self.m_tokens_in

        self.Backbone_C = Transformer(input_dim=self.M_C.output_dim, output_dim=self.C_encoding_dim, m_tokens=self.m_tokens_in, cfg=self.bert_cfg)
        self.Backbone_V = Transformer(input_dim=self.M_V.output_dim+self.C_encoding_dim, output_dim=self.set_encoding_dim, m_tokens=self.m_tokens_in, cfg=self.bert_cfg)
        self.fc = nn.Linear(self.set_encoding_dim*self.m_tokens_in, 128, bias=True)
        self.out_net = nn.Linear(128, self.output_dim, bias=True)

    def forward(self, V, C_d):
        batchsize = V.shape[0]
        # V: batch, nVar, dim_V; C_d: batch, nVar, dim_C
        C = self.M_C(C_d)
        C = self.Backbone_C(C)
        C = gelu(C)
        V = self.M_V(V)
        V_C = torch.cat((V,C),2)
        V_C = self.Backbone_V(V_C)
        V_C = gelu(V_C)
        V_C = self.fc(V_C.view(batchsize,-1))
        V_C = gelu(V_C)
        output = self.out_net(V_C)

        return output


class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""

    def __init__(self, input_dim, output_dim, m_tokens, cfg=Config()):
        super().__init__()
        self.in_net = nn.Linear(input_dim, cfg.dim, bias=True)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.m_tokens = m_tokens
        self.out_net = nn.Linear(in_features=cfg.dim, out_features=output_dim, bias=True)

    def forward(self, x):
        # shape x: batch_size x m_token x m_state
        h = self.in_net(x)
        for block in self.blocks:
            h = block(h, None)
        y = self.out_net(h)
        return y