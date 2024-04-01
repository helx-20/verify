import torch
import torch.nn as nn
from .bert import Embeddings, Block, Config
import math
import torch.nn.functional as F

surrograte_type = 'G'

gamma = 0.5
lens = 0.5
R_m = 1
beta_value = 1.8
b_j0_value = 0.01
def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma

class Critical(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # judge if critical
        ctx.save_for_backward(input)
        return input.gt(0).float()
    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        scale = 6.0
        hight = .15
        if surrograte_type == 'G':
            temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        elif surrograte_type == 'MG':
            temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
                - gaussian(input, mu=lens, sigma=scale * lens) * hight \
                - gaussian(input, mu=-lens, sigma=scale * lens) * hight
        elif surrograte_type =='linear':
            temp = F.relu(1-input.abs())
        elif surrograte_type == 'slayer':
            temp = torch.exp(-5*input.abs())
        elif surrograte_type == 'rect':
            temp = abs(input) < lens
        return grad_input * temp.float()

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

def define_critical_net(model, input_dim, output_dim, m_tokens_in, m_tokens_out, transformer_out_feature_dim=256):

    h_dim = 256

    # define input positional mapping
    M = PositionalMapping(input_dim=input_dim, L=4, pos_scale=0.01, heading_scale=0.01)

    # define backbone networks
    if model == 'transformer':
        bert_cfg = Config() 
        bert_cfg.dim = h_dim
        bert_cfg.n_layers = 4
        bert_cfg.n_heads = 4
        bert_cfg.max_len = m_tokens_in
        Backbone = Transformer(input_dim=M.output_dim, output_dim=transformer_out_feature_dim, m_tokens=m_tokens_in, cfg=bert_cfg)
    else:
        raise NotImplementedError(
            'Wrong backbone model name %s (choose one from [transformer])' % model)

    output_net = Out_net(m_tokens_in*transformer_out_feature_dim, output_dim, m_tokens_out)

    return nn.Sequential(M, Backbone, output_net)


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

class Out_net(nn.Module):
    def __init__(self, input_dim, output_dim, m_tokens):
        super().__init__()
        self.output_dim = output_dim
        self.m_tokens = m_tokens
        self.mask_net = nn.Linear(input_dim, m_tokens, bias=True)
        self.output_set_net = nn.Linear(input_dim, m_tokens*output_dim, bias=True)
        self.mask_thr = 0.5
        self.judge_critical = Critical.apply 

    def forward(self, x):
        bs = x.shape[0]
        x = x.view(bs, -1)
        #mask = self.mask_net(x).view(bs, self.m_tokens, 1)
        mask = self.judge_critical(self.mask_net(x).view(bs, self.m_tokens, 1) - self.mask_thr)
        output_set = self.output_set_net(x).view(bs, self.m_tokens, self.output_dim)
        return mask, output_set * mask