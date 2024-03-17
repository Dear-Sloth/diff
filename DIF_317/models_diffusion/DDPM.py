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

from models_diffusion.DDPM_modules.DDPM_UVIT import *
from models_diffusion.diffusion_worker import *


class Model(nn.Module):
    
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args
        self.device = args.device
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.input_size = args.num_vars
        self.diff_steps = args.diff_steps
        self.stride=args.stride
        self.patch_len=args.patch_len
        self.embed_dim=args.embed_dim
        self.depth=args.depth
        self.num_heads=args.num_heads
        self.mlp_dim = args.mlp_dim
        if args.UNet_Type == "VIT":
            u_net = UViT(args,context_window=self.seq_len, 
                         target_window=self.pred_len, 
                         stride=self.stride, 
                         patch_len=self.patch_len, 
                         embed_dim=self.embed_dim, 
                         depth=self.depth, 
                         num_heads=self.num_heads, 
                         mlp_time_embed=True,
                         mlp_dim=self.mlp_dim,
                         dropout=self.args.dropout)
        self.u_net = u_net
        self.diffusion_worker = Diffusion_Worker(args, u_net, self.diff_steps)
        self.short_term_range = args.seq_len 


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, sample_times=5):
        if self.training:
            return self.forward_train(x_enc, x_mark_enc, x_dec, x_mark_dec,
                                             enc_self_mask, dec_self_mask, dec_enc_mask)
        else:
            return self.forward_val_test(x_enc, x_mark_enc, x_dec, x_mark_dec,
                                            enc_self_mask, dec_self_mask, dec_enc_mask, sample_times)


    def forward_train(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        self.diffusion_worker.train()
        #print(np.shape(x_enc),np.shape(x_dec))  (B,L,N)
        x_enc_i = x_enc
        x_dec_i = x_dec
        x_future = x_dec_i[:,-self.args.pred_len:,:] 
        x_past = x_enc_i.permute(0,2,1)     
        x_future = x_future.permute(0,2,1) 
        f_dim = -1 if self.args.features in ['MS'] else 0
        loss = self.diffusion_worker(x_future[:,f_dim:,:],x_past)
        return loss

    def forward_val_test(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, sample_times=5):

        self.diffusion_worker.eval()

        x_enc_i = x_enc
        x_dec_i = x_dec

        x_past = x_enc_i
        x_future = x_dec_i[:,-self.args.pred_len:,:]

        x_past = x_past.permute(0,2,1)     
        x_future = x_future.permute(0,2,1) #(B,N,L)

        f_dim = -1 if self.args.features in ['MS'] else 0

        B, nF, nL = np.shape(x_past)[0], self.input_size, self.pred_len
        B = B*self.input_size
        if self.args.features in ['MS']:
            nF = 1
        shape = [nF, nL]
        
        all_outs = []
        for i in range(sample_times):
            start_code = torch.randn((B, nL), device='cuda:0')
            samples_ddim = self.diffusion_worker.sample_ddim(S=20,
                                             conditioning=x_past,
                                             batch_size=B,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=1.0,
                                             unconditional_conditioning=None,
                                             eta=0.,
                                             x_T=start_code)
            samples_ddim=samples_ddim.permute(0,2,1)
            all_outs.append(samples_ddim)
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




