import torch
import torch.nn as nn
import math
from .timm import trunc_normal_, Mlp
import einops
import torch.utils.checkpoint
import numpy as np
import torch.nn.functional as F

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')


def patchify(x,patch_len,stride):
    
    padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
    
    #print("1",x.shape)
    x = padding_patch_layer(x)
    #print("2",x.shape)
    x = x.unfold(dimension=-1, size=patch_len, step=stride)   # x: [bs x patch_num x patch_len]
    #print("3",x.shape)
    return x


class Flatten_Head(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                               # x: [bs x d_model x patch_num]  
        x = self.flatten(x)                             # z: [bs x d_model * patch_num]
        x = self.linear(x)                              # z: [bs x target_window]
        x = self.dropout(x)
        return x
        
def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class DiffEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):

        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x
    
    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0) 
        table = steps * frequencies 
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  
        return table

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)

class Block(nn.Module):

    def __init__(self, dim, num_heads,dropout, mlp_dim=256, qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,proj_drop=dropout)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(mlp_dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,drop=dropout)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint
        self.dropout = nn.Dropout(dropout)
        self.norm_batch = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(dim), Transpose(1,2))
    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.dropout(self.attn(x))
        x = x + self.mlp(self.norm2(x))
        x = self.norm1(x)
        return x

    

class UViT(nn.Module):
    def __init__(self, args,context_window, target_window, dropout,stride, patch_len, embed_dim, depth, num_heads, mlp_dim=256,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=True,
                 use_checkpoint=False, num_pred_patch=0, skip=True,):
        super().__init__()
        self.args = args
        self.skip_drop= nn.Dropout(self.args.skip_dropout)
        self.data_drop= nn.Dropout(self.args.data_dropout)
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = nn.Linear(patch_len, embed_dim)
        self.stride = stride
        self.patch_len = patch_len
        self.context_window = context_window
        self.target_window = target_window
        
        if self.args.test_vit == True:
            num_patches = int((context_window - patch_len) / stride + 1) + 1
            self.total_token = num_patches
        else:
            num_patches = int((context_window - patch_len) / stride + 1) + 1
            num_pred_patch=int((target_window - patch_len) / stride + 1) + 1
            self.total_token = num_patches + num_pred_patch + 1

        self.final_layer = Flatten_Head(embed_dim*self.total_token, target_window)
        

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            #nn.SiLU(), 
        ) if mlp_time_embed else nn.Identity()

        self.timembedding = DiffEmbedding(
            num_steps=self.args.diff_steps,
            embedding_dim=self.embed_dim,
        )

        self.act = lambda x: x * torch.sigmoid(x)

        self.cond_embed = nn.Linear(patch_len, embed_dim)

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint,dropout=dropout)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint,dropout=dropout)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint,dropout=dropout)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.total_token, embed_dim))
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x, timesteps=None, cond_ts=None):

        x = patchify(x,self.patch_len,self.stride)
        x = self.patch_embed(x)
        if not self.args.test_vit:
            cond_ts = patchify(cond_ts,self.patch_len,self.stride)
            cond_token = self.cond_embed(cond_ts)
            cond_token = self.data_drop(cond_token)
            time_token = self.act(self.timembedding(timesteps.long()))
            time_token = time_token.unsqueeze(dim=1)
            #print(f'pos_embed shape: {self.pos_embed.shape}, time_token shape: {time_token.shape}, cond_token shape: {cond_token.shape}, x shape: {x.shape}')

            x = torch.cat((time_token, cond_token, x), dim=1)
        else:
            pass
        

        x = x +  self.pos_embed
        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)
        x = self.mid_block(x)
        for blk in self.out_blocks:
            x = blk(x, self.skip_drop(skips.pop()))
        x = self.norm(x)
        x = self.final_layer(x)
        return x
