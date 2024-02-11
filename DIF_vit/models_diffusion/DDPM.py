

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
        self.stride=args.stride
        self.patch_len=args.patch_len
        self.embed_dim=args.embed_dim
        self.depth=args.depth
        self.num_heads=args.num_heads
        self.conditional = args.conditional
        self.mlp_ratio = args.mlp_ratio
        if args.UNet_Type == "VIT":
            u_net = UViT(args,context_window=self.seq_len, 
                         target_window=self.pred_len, 
                         stride=self.stride, 
                         patch_len=self.patch_len, 
                         embed_dim=self.embed_dim, 
                         depth=self.depth, 
                         num_heads=self.num_heads, 
                         mlp_time_embed=True,
                         cond=self.conditional,
                         mlp_ratio=self.mlp_ratio,)
        self.u_net = u_net
        self.diffusion_worker = Diffusion_Worker(args, u_net, self.diff_steps)

        if args.type_sampler == "none":
            pass
        elif args.type_sampler == "dpm":
            assert self.args.parameterization == "x_start"
            self.sampler = DPMSolverSampler(args,u_net, self.diffusion_worker)
        self.short_term_range = args.seq_len 
        self.norm_len = args.label_len


    def train_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        self.diffusion_worker.train()
       
        if self.args.use_window_normalization:
            seq_len = np.shape(x_enc)[1]
            mean_ = torch.mean(x_enc[:,-self.norm_len:,:], dim=1).unsqueeze(1)
            std_ = torch.ones_like(torch.std(x_enc, dim=1).unsqueeze(1))
            x_enc_i = (x_enc-mean_.repeat(1,seq_len,1))/(std_.repeat(1,seq_len,1)+0.00001)
            seq_len = np.shape(x_dec)[1]
            x_dec_i = (x_dec-mean_.repeat(1,seq_len,1))/(std_.repeat(1,seq_len,1)+0.00001)
        else:
            x_enc_i = x_enc
            x_dec_i = x_dec


        x_future = x_dec_i[:,-self.args.pred_len:,:] 
        x_past = x_enc_i.permute(0,2,1)     
        x_future = x_future.permute(0,2,1) 

        f_dim = -1 if self.args.features in ['MS'] else 0

        a1,b1,c1 = x_future[:,f_dim:,:].shape
        a2,b2,c2 = x_past.shape
        xf_ind = torch.reshape(x_future[:,f_dim:,:],(a1*b1,c1))
        xp_ind = torch.reshape(x_past,(a2*b2,c2))
        loss = self.diffusion_worker(xf_ind, xp_ind)
        return loss

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, sample_times=5):

        self.diffusion_worker.eval()
        
        if self.args.use_window_normalization:
            seq_len = np.shape(x_enc)[1]
            mean_ = torch.mean(x_enc[:,-self.norm_len:,:], dim=1).unsqueeze(1)
            std_ = torch.ones_like(torch.std(x_enc, dim=1).unsqueeze(1))
            x_enc_i = (x_enc-mean_.repeat(1,seq_len,1))/(std_.repeat(1,seq_len,1)+0.00001)
            seq_len = np.shape(x_dec)[1]
            x_dec_i = (x_dec-mean_.repeat(1,seq_len,1))/(std_.repeat(1,seq_len,1)+0.00001)               
        else:
            x_enc_i = x_enc
            x_dec_i = x_dec

        x_past = x_enc_i
        x_future = x_dec_i[:,-self.args.pred_len:,:]

        x_past = x_past.permute(0,2,1)     
        x_future = x_future.permute(0,2,1) 

        f_dim = -1 if self.args.features in ['MS'] else 0
        a1,b1,c1 = x_future[:,f_dim:,:].shape
        a2,b2,c2 = x_past.shape
        xf_ind = torch.reshape(x_future[:,f_dim:,:],(a1*b1,c1))
        xp_ind = torch.reshape(x_past,(a2*b2,c2))   

        B, nF, nL = np.shape(xp_ind)[0], 1, self.pred_len
        if self.args.features in ['MS']:
            nF = 1
        shape = [nF, nL]
        
        all_outs = []
        for i in range(sample_times):
            start_code = torch.randn((B, nL), device=self.device)
       
            if self.conditional:
                samples_ddim, _ = self.sampler.sample(S=20,
                                             conditioning=xp_ind,
                                             batch_size=B,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=1.0,
                                             unconditional_conditioning=None,
                                             eta=0.,
                                             x_T=start_code)
                #print(samples_ddim.shape)
                samples_ddim = torch.reshape(samples_ddim,(a1,b1,c1))
                outs_i = samples_ddim.permute(0,2,1)
                # print(outs_i.shape)torch.Size([64, 192, 7])
            else:
                samples_ddim = self.u_net(xp_ind)
                #print(samples_ddim.shape)
                samples_ddim = torch.reshape(samples_ddim,(a1,b1,c1))
                outs_i = samples_ddim.permute(0,2,1)

            if self.args.use_window_normalization:
                out_len = np.shape(outs_i)[1]
                outs_i = outs_i * std_.repeat(1,out_len,1) + mean_.repeat(1,out_len,1)

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

        if self.args.use_window_normalization:
            
            out_len = np.shape(x_enc_i)[1]
            x_enc_i = x_enc_i * std_.repeat(1,out_len,1) + mean_.repeat(1,out_len,1)

            out_len = np.shape(x_dec_i)[1]
            x_dec_i = x_dec_i * std_.repeat(1,out_len,1) + mean_.repeat(1,out_len,1)


        return outs, x_enc[:,:,f_dim:], x_dec[:, -self.args.pred_len:, f_dim:], None, None




