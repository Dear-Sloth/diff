

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
from .samplers.dpm_sampler import DPMSolverSampler
from utils.diffusion_utils import *
from models_diffusion.DDPM_modules.RevIN import RevIN

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


class Diffusion_Worker(nn.Module):
    
    def __init__(self, args, u_net=None, diff_steps=1000):
        super(Diffusion_Worker, self).__init__()

        self.args = args
        self.device = args.device
        self.pred_len = args.pred_len
        self.num_vars = args.num_vars
        self.batch_size =args.batch_size
        self.parameterization = args.parameterization
        assert self.parameterization in ["noise", "x_start"], 'currently only supporting "eps" and "x0"'
        self.conditional = args.conditional
        self.diff_train_steps = diff_steps
        self.diff_test_steps = diff_steps
        
        self.beta_start = 1e-4 # 1e4
        self.beta_end = 2e-2
        self.beta_schedule = "cosine"

        self.v_posterior = 0.0
        self.original_elbo_weight = 0.0
        self.l_simple_weight = 1

        self.loss_type = "l2"

        self.set_new_noise_schedule(None, self.beta_schedule, self.diff_train_steps, self.beta_start, self.beta_end)

        self.clip_denoised = True

        self.total_N = len(self.alphas_cumprod)
        self.T = 1.
        self.eps = 1e-5
        self.nn = u_net
        self.latent_dim = args.latent_dim
        if args.type_sampler == "none":
            pass
        elif args.type_sampler == "dpm":
            assert self.args.parameterization == "x_start"
            self.sampler = DPMSolverSampler(args,self.nn, self.device,self.alphas_cumprod,self.betas.device)

        if self.args.revinnorm: self.revin_layer = RevIN(self.num_vars, affine=True, subtract_last=False)

        self.cnnenc = torch.nn.Conv1d(in_channels = self.args.seq_len*self.args.num_vars,
                                          out_channels = self.args.pred_len*self.args.num_vars,
                                          kernel_size=1,
                                          groups = self.args.num_vars)
        self.linearproj= nn.Linear(self.args.seq_len,self.args.pred_len)
        self.projtype='CNN'

    def set_new_noise_schedule(self, given_betas=None, beta_schedule="linear", diff_steps=1000, beta_start=1e-4, beta_end=2e-2
    ):  
        if exists(given_betas):
            betas = given_betas
        else:
            if beta_schedule == "linear":
                betas = np.linspace(beta_start, beta_end, diff_steps)
            elif beta_schedule == "quad":
                betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, diff_steps) ** 2
            elif beta_schedule == "const":
                betas = beta_end * np.ones(diff_steps)
            elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
                betas = 1.0 / np.linspace(diff_steps, 1, diff_steps)
            elif beta_schedule == "sigmoid":
                betas = np.linspace(-6, 6, diff_steps)
                betas = (beta_end - beta_start) / (np.exp(-betas) + 1) + beta_start
            elif beta_schedule == "cosine":
                betas = cosine_beta_schedule(diff_steps)
            else:
                raise NotImplementedError(beta_schedule)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = beta_start
        self.linear_end = beta_end

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "noise":
            lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x_start":
            lvlb_weights = 0.8 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")

        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss  

    def noise_ts(self, x_start, t, noise=None):

        noise = default(noise, lambda: self.scaling_noise * torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def forward(self, x=None, cond_ts=None):

        # Feed inputs ith shape of [bsz, feas, seqlen]
        # both x and cond are two time seires 
        if self.conditional:
            cond_ts = cond_ts.permute(0,2,1) #(B,L,N)
            cond_ts=self.revin_layer(cond_ts,'norm')
            cond_ts = cond_ts.permute(0,2,1) #(B,N,L)
            B = np.shape(x)[0]
            N = self.args.num_vars
            L1 = np.shape(cond_ts)[2]
            L2 = np.shape(x)[2]
            cond_ts = torch.reshape(cond_ts,(B*N,L1))
            x = torch.reshape(x,(B*N,L2))
            t = torch.randint(0, self.num_timesteps, size=[B*N//2,]).long().to(self.device)
            t = torch.cat([t, self.num_timesteps-1-t], dim=0)
            noise = torch.randn_like(x)
            x_k = self.noise_ts(x_start=x, t=t, noise=noise)
            model_out= self.nn(x_k, t, cond_ts)
            model_out=torch.reshape(model_out,(B,N,L2))
            model_out = model_out.permute(0,2,1) #(B,TARGET L,N)
            model_out=self.revin_layer(model_out,'denorm')
            model_out = model_out.permute(0,2,1)  #(B,N,TARGET L)
            x = torch.reshape(x,(B,N,L2))
            
        else:
            B = np.shape(cond_ts)[0]
            N = self.args.num_vars
            cond_ts = cond_ts.permute(0,2,1) #(B,L,N)
            cond_ts=self.revin_layer(cond_ts,'norm')
            cond_ts = cond_ts.permute(0,2,1) #(B,N,L)
            if self.projtype=='linear':
                cond_ts = torch.reshape(cond_ts,(B*N,-1))
                cond_ts = self.linearproj(cond_ts)
                L = np.shape(cond_ts)[1]
            else:
                cond_ts=self.cnnenc(cond_ts.reshape(B,-1).unsqueeze(-1))    #(B,N,TARGET L)
                cond_ts=cond_ts.squeeze(-1).reshape(B,N,-1)            
                L = np.shape(cond_ts)[2]
                cond_ts = torch.reshape(cond_ts,(B*N,L))
            #noise = torch.randn_like(cond_ts)
            #x_k = self.noise_ts(x_start=cond_ts, t=t, noise=noise)
            t = torch.randint(0, self.num_timesteps, size=[(B*N)//2,]).long().to(self.device)
            t = torch.cat([t, self.num_timesteps-1-t], dim=0)           
            model_out = self.nn(cond_ts,t,None)
            model_out = torch.reshape(model_out,(B,N,L))
            model_out = model_out.permute(0,2,1) #(B,TARGET L,N)
            model_out = self.revin_layer(model_out,'denorm')
            model_out = model_out.permute(0,2,1)  #(B,N,TARGET L)
        if self.parameterization == "noise":
            target = noise 
        elif self.parameterization == "x_start":
            target = x
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")
        
        # in the submission version, we calculate time first, and then calculate variable
        f_dim = -1 if self.args.features == 'MS' else 0
        bias_weight=False
        if bias_weight:
            model_out =torch.reshape(model_out,(B*N,-1))
            target =torch.reshape(target,(B*N,-1))
            loss = self.get_loss(model_out[:,:], target[:,:], mean=False).mean(dim=1)
            loss_simple = loss.mean() * self.l_simple_weight
            loss_vlb = (self.lvlb_weights[t] * loss).mean()
            loss = loss_simple + self.original_elbo_weight * loss_vlb
        else:
            loss = self.get_loss(model_out[:,f_dim:,:], target[:,f_dim:,:], mean=False).mean(dim=1).mean(dim=1).mean()
        return loss

    def sample_ddim(self,S,conditioning,batch_size,shape,x_T,verbose=False,unconditional_guidance_scale=1.0,unconditional_conditioning=None, eta=0.):
        if self.conditional:
            B = np.shape(conditioning)[0]
            N = self.args.num_vars
            conditioning = conditioning.permute(0,2,1) #(B,L,N)
            conditioning = self.revin_layer(conditioning,'norm')
            conditioning = conditioning.permute(0,2,1) #(B,N,L)
            conditioning = torch.reshape(conditioning,(B*N,-1))
            samples_ddim ,_= self.sampler.sample(S=20,
                                             conditioning=conditioning,
                                             batch_size=batch_size,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=1.0,
                                             unconditional_conditioning=None,
                                             eta=0.,
                                             x_T=x_T)
            samples_ddim=torch.reshape(samples_ddim,(B,N,-1))                                
            samples_ddim = samples_ddim.permute(0,2,1).to(self.device) #(B,TARGET L,N)
            samples_ddim = self.revin_layer(samples_ddim,'denorm')
            samples_ddim = samples_ddim.permute(0,2,1)
        else:
            x_T = x_T.permute(0,2,1) #(B,L,N)
            x_T=self.revin_layer(x_T,'norm')
            x_T = x_T.permute(0,2,1) #(B,N,L)
            B = np.shape(x_T)[0]
            N = self.args.num_vars
            if self.projtype=='linear':
                x_T = torch.reshape(x_T,(B*N,-1))
                x_T = self.linearproj(x_T)
                L = np.shape(x_T)[1]
            else:
                x_T=self.cnnenc(x_T.reshape(B,-1).unsqueeze(-1))   
                x_T=x_T.squeeze(-1).reshape(B,N,-1)      #(B,N,TARGET L)
                L = np.shape(x_T)[2]
                x_T = torch.reshape(x_T,(B*N,L))
            #noise = torch.randn_like(x_T)
            #x_k = self.noise_ts(x_start=x_T, t=t, noise=noise)
            samples_ddim ,_= self.sampler.sample(S=20,
                                             conditioning=conditioning,
                                             batch_size=batch_size,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=1.0,
                                             unconditional_conditioning=None,
                                             eta=0.,
                                             x_T=x_T)
            samples_ddim=torch.reshape(samples_ddim,(B,N,L))                                
            samples_ddim = samples_ddim.permute(0,2,1).to(self.device) #(B,TARGET L,N)
            samples_ddim=self.revin_layer(samples_ddim,'denorm')
            samples_ddim = samples_ddim.permute(0,2,1)  #(B,N,TARGET L)
                   
        return samples_ddim


        



