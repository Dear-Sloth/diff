

from typing import List, Optional, Tuple, Union
from typing import Any, Dict
from functools import partial
from inspect import isfunction
from tqdm import tqdm
import math
import torch
import torch.nn as nn
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn.modules import loss
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from models_diffusion.DDPM_modules.PatchTST_backbone import PatchTST_backbone
from models_diffusion.diffusion_worker import *

from .samplers.dpm_sampler import DPMSolverSampler


class Model(nn.Module):
    
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args
        self.device = args.device
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.input_size = args.num_vars
        self.diff_steps = args.diff_steps
        self.conditional = args.conditional
        self.c_in =args.num_vars
        if self.conditional:
            self.context_window =args.seq_len + args.pred_len
        else:
            self.context_window =args.seq_len
        self.target_window =args.pred_len
        self.n_layers =args.e_layers
        self.n_heads =args.n_heads
        self.d_model =args.d_model
        self.d_ff =args.d_ff
        self.dropout =args.dropout
        self.fc_dropout =args.fc_dropout
        self.head_dropout =args.head_dropout
        self.individual =args.individual
        self.patch_len =args.patch_len
        self.stride =args.stride
        self.padding_patch =args.padding_patch
        self.revin =args.revin
        self.affine =args.affine
        self.subtract_last =args.subtract_last
        self.kernel_size =args.kernel_size

        if args.UNet_Type == "TST":
            u_net = PatchTST_backbone(conditional=self.conditional,c_in=self.c_in, context_window = self.context_window, target_window=self.target_window, patch_len=self.patch_len, stride=self.stride, 
                                   n_layers=self.n_layers, d_model=self.d_model,
                                  n_heads=self.n_heads, d_k=None, d_v=None, d_ff=self.d_ff, norm='BatchNorm', attn_dropout=0.,
                                  dropout=self.dropout, act="gelu", key_padding_mask='auto', padding_var=None, 
                                  attn_mask=None, res_attention=True, pre_norm=False, store_attn=False,
                                  pe='zeros', learn_pe=True, fc_dropout=self.fc_dropout, head_dropout=self.head_dropout, padding_patch = self.padding_patch,
                                  pretrain_head=False, head_type='flatten', individual=self.individual, revin=self.revin, affine=self.affine,
                                  subtract_last=self.subtract_last, verbose=False)
        self.u_net = u_net
        self.diffusion_worker = Diffusion_Worker(args, u_net, self.diff_steps)

        if args.type_sampler == "none":
            pass
        elif args.type_sampler == "dpm":
            assert self.args.parameterization == "x_start"
            self.sampler = DPMSolverSampler(args,u_net, self.diffusion_worker)


    def train_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        self.diffusion_worker.train()
 
        x_future = x_dec[:,-self.args.pred_len:,:] 
        x_past = x_enc.permute(0,2,1)     
        x_future = x_future.permute(0,2,1) 

        f_dim = -1 if self.args.features in ['MS'] else 0
        loss = self.diffusion_worker(x_future[:,f_dim:,:], x_past)  #(B,C,L)
        return loss

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, sample_times=5):

        self.diffusion_worker.eval()
        
        x_past = x_enc
        x_future = x_dec[:,-self.args.pred_len:,:]
        x_past = x_past.permute(0,2,1)     
        x_future = x_future.permute(0,2,1) 
        f_dim = -1 if self.args.features in ['MS'] else 0

        B, nF, nL = np.shape(x_past)[0], np.shape(x_past)[1], self.pred_len
        if self.args.features in ['MS']:
            nF = 1
        shape = [nF, nL]
        
        all_outs = []
        for i in range(sample_times):
            start_code = torch.randn((B, nF, nL), device=self.device)
       
            if self.conditional:
                samples_ddim, _ = self.sampler.sample(S=20,
                                             conditioning=x_past,
                                             batch_size=B,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=1.0,
                                             unconditional_conditioning=None,
                                             eta=0.,
                                             x_T=start_code)
                #print(samples_ddim.shape)
                outs_i = samples_ddim.permute(0,2,1)
                # print(outs_i.shape)torch.Size([64, 192, 7])
            else:
                samples_ddim = self.u_net(x_past)
                #print(samples_ddim.shape)
                outs_i = samples_ddim.permute(0,2,1)

            all_outs.append(outs_i)
        all_outs = torch.stack(all_outs, dim=0)

        flag_return_all = True

        if flag_return_all:
            outs = all_outs.permute(1,0,2,3)
            f_dim = -1 if self.args.features in ['MS'] else 0
            outs = outs[:, :, -self.pred_len:, f_dim:] # - 0.4
        else:
            outs = all_outs.mean(0)
            f_dim = -1 if self.args.features == ['MS'] else 0
            outs = outs[:, -self.pred_len:, f_dim:] # - 0.4

        return outs, x_enc[:,:,f_dim:], x_dec[:, -self.args.pred_len:, f_dim:], None, None




