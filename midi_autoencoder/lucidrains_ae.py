import math
from functools import partial, wraps

import torch
from torch import sqrt
from torch import nn, einsum
import torch.nn.functional as F
from torch.special import expm1
from torch.amp import autocast

from tqdm import tqdm
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def identity(t):
    return t

def is_lambda(f):
    return callable(f) and f.__name__ == "<lambda>"

def default(val, d):
    if exists(val):
        return val
    return d() if is_lambda(d) else d

def cast_tuple(t, l = 1):
    return ((t,) * l) if not isinstance(t, tuple) else t

def append_dims(t, dims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * dims))

def l2norm(t):
    return F.normalize(t, dim = -1)

# u-vit related functions and modules

class Upsample(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        factor = 2
    ):
        super().__init__()
        self.factor = factor
        self.factor_squared = factor ** 2

        dim_out = default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * self.factor_squared, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(factor)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // self.factor_squared, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r = self.factor_squared)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)

def Downsample(
    dim,
    dim_out = None,
    factor = 2
):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = factor, p2 = factor),
        nn.Conv2d(dim * (factor ** 2), default(dim_out, dim), 1)
    )

class RMSNorm(nn.Module):
    def __init__(self, dim, scale = True, normalize_dim = 2):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim)) if scale else 1

        self.scale = scale
        self.normalize_dim = normalize_dim

    def forward(self, x):
        normalize_dim = self.normalize_dim
        scale = append_dims(self.g, x.ndim - self.normalize_dim - 1) if self.scale else 1
        return F.normalize(x, dim = normalize_dim) * scale * (x.shape[normalize_dim] ** 0.5)

# sinusoidal positional embeds

class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out, normalize_dim = 1)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim, normalize_dim = 1)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim, normalize_dim = 1)
        )

    def forward(self, x):
        residual = x

        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)

        return self.to_out(out) + residual

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, scale = 8, dropout = 0.):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        q, k = map(l2norm, (q, k))

        q = q * self.q_scale
        k = k * self.k_scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.norm = RMSNorm(dim, scale = False)
        dim_hidden = dim * mult

        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim_hidden * 2),
            Rearrange('b d -> b 1 d')
        )

        to_scale_shift_linear = self.to_scale_shift[-2]
        nn.init.zeros_(to_scale_shift_linear.weight)
        nn.init.zeros_(to_scale_shift_linear.bias)

        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim_hidden, bias = False),
            nn.SiLU()
        )

        self.proj_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim, bias = False)
        )

    def forward(self, x, t):
        x = self.norm(x)
        x = self.proj_in(x)

        scale, shift = self.to_scale_shift(t).chunk(2, dim = -1)
        x = x * (scale + 1) + shift

        return self.proj_out(x)

# vit

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        time_cond_dim,
        depth,
        dim_head = 32,
        heads = 4,
        ff_mult = 4,
        dropout = 0.,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = dropout),
                FeedForward(dim = dim, mult = ff_mult, cond_dim = time_cond_dim, dropout = dropout)
            ]))

    def forward(self, x, t):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x, t) + x

        return x

# 1D Modules

class Upsample1d(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        factor = 2
    ):
        super().__init__()
        self.factor = factor
        self.factor_squared = factor

        dim_out = default(dim_out, dim)
        conv = nn.Conv1d(dim, dim_out * self.factor, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            Rearrange('b (c r) l -> b c (l r)', r = factor)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, w = conv.weight.shape
        conv_weight = torch.empty(o // self.factor, i, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r = self.factor)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)

def Downsample1d(
    dim,
    dim_out = None,
    factor = 2
):
    return nn.Sequential(
        Rearrange('b c (l p) -> b (c p) l', p = factor),
        nn.Conv1d(dim * factor, default(dim_out, dim), 1)
    )

class Block1d(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out, normalize_dim = 1)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock1d(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block1d(dim, dim_out)
        self.block2 = Block1d(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention1d(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim, normalize_dim = 1)
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim, normalize_dim = 1)
        )

    def forward(self, x):
        residual = x

        b, c, l = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x -> b h c x', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c x -> b (h c) x', h = self.heads)

        return self.to_out(out) + residual

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

# diffusion helpers

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# logsnr schedules and shifting / interpolating decorators
# only cosine for now

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def logsnr_schedule_cosine(t, logsnr_min = -15, logsnr_max = 15):
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * log(torch.tan(t_min + t * (t_max - t_min)))

def logsnr_schedule_shifted(fn, image_d, noise_d):
    shift = 2 * math.log(noise_d / image_d)
    @wraps(fn)
    def inner(*args, **kwargs):
        nonlocal shift
        return fn(*args, **kwargs) + shift
    return inner

def logsnr_schedule_interpolated(fn, image_d, noise_d_low, noise_d_high):
    logsnr_low_fn = logsnr_schedule_shifted(fn, image_d, noise_d_low)
    logsnr_high_fn = logsnr_schedule_shifted(fn, image_d, noise_d_high)

    @wraps(fn)
    def inner(t, *args, **kwargs):
        nonlocal logsnr_low_fn
        nonlocal logsnr_high_fn
        return t * logsnr_low_fn(t, *args, **kwargs) + (1 - t) * logsnr_high_fn(t, *args, **kwargs)

    return inner


class UNetStyleVAE(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        ch=64,
        ch_mult=(1, 2, 4, 4),
        ch_mult_1d=None,
        num_res_blocks=2,
        z_channels=16,
        kl_weight=1e-6,
        binary_mode=False,
        attn_heads=4,
        attn_dim_head=32,
        loss_type='bce',
        focal_gamma=2.0,
        focal_alpha=0.25
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.binary_mode = binary_mode
        self.z_channels = z_channels
        self.ch_mult_1d = ch_mult_1d
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        # --- Encoder ---
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, ch, 3, padding=1)
        
        dims = [ch] + [ch * m for m in ch_mult]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        self.downs = nn.ModuleList([])
        
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (self.num_resolutions - 1)
            
            blocks = nn.ModuleList([])
            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(dim_in, dim_in, time_emb_dim=None))
            
            attn = LinearAttention(dim_in, heads=attn_heads, dim_head=attn_dim_head)
            
            if not is_last:
                down = Downsample(dim_in, dim_out)
            else:
                # Last level, just change channels without downsampling
                down = nn.Conv2d(dim_in, dim_out, 3, padding=1)
            
            self.downs.append(nn.ModuleList([
                blocks,
                attn,
                down
            ]))

        mid_dim = dims[-1]
        
        # 1D Branch
        self.downs_1d = nn.ModuleList([])
        self.ups_1d = nn.ModuleList([])
        
        if ch_mult_1d is not None:
            assert len(ch_mult) == 4, "ch_mult must downsample by a factor of 8"
            # Collapse to 1D using a bottleneck Conv2d
            # (B, C, T, 11) -> (B, C/4, T, 11) -> (B, C, T)
            self.to_1d = nn.Sequential(
                nn.Conv2d(mid_dim, mid_dim // 4, kernel_size=1),
                nn.Conv2d(mid_dim // 4, mid_dim, kernel_size=(1, 11), padding=0),
                Rearrange('b c t 1 -> b c t')
            )
            
            # Expand from 1D using PixelShuffle-like operation
            # (B, C, T) -> (B, (C/4)*11, T) -> (B, C/4, T, 11) -> (B, C, T, 11)
            self.from_1d = nn.Sequential(
                nn.Conv1d(mid_dim, (mid_dim // 4) * 11, kernel_size=1),
                Rearrange('b (c p) t -> b c t p', p=11),
                nn.Conv2d(mid_dim // 4, mid_dim, kernel_size=1)
            )
            
            dims_1d = [mid_dim] + [ch * m for m in ch_mult_1d]
            in_out_1d = list(zip(dims_1d[:-1], dims_1d[1:]))
            
            for ind, (dim_in, dim_out) in enumerate(in_out_1d):
                is_last = ind >= (len(in_out_1d) - 1)
                
                blocks = nn.ModuleList([])
                for _ in range(num_res_blocks):
                    blocks.append(ResnetBlock1d(dim_in, dim_in, time_emb_dim=None))
                
                attn = LinearAttention1d(dim_in, heads=attn_heads, dim_head=attn_dim_head)
                
                if ind == 0:
                    down = nn.Conv1d(dim_in, dim_out, 3, padding=1)
                else:
                    down = Downsample1d(dim_in, dim_out)
                
                self.downs_1d.append(nn.ModuleList([
                    blocks,
                    attn,
                    down
                ]))
                
            mid_dim = dims_1d[-1]
            
            # 1D Bottleneck
            self.mid_block1 = ResnetBlock1d(mid_dim, mid_dim, time_emb_dim=None)
            self.mid_attn = LinearAttention1d(mid_dim, heads=attn_heads, dim_head=attn_dim_head)
            self.mid_block2 = ResnetBlock1d(mid_dim, mid_dim, time_emb_dim=None)
            
            self.norm_out = RMSNorm(mid_dim, normalize_dim=1)
            self.to_z = nn.Conv1d(mid_dim, 2 * z_channels, 1)
            self.from_z = nn.Conv1d(z_channels, mid_dim, 1)
            
            self.mid_dec_block1 = ResnetBlock1d(mid_dim, mid_dim, time_emb_dim=None)
            self.mid_dec_attn = LinearAttention1d(mid_dim, heads=attn_heads, dim_head=attn_dim_head)
            self.mid_dec_block2 = ResnetBlock1d(mid_dim, mid_dim, time_emb_dim=None)
            
            # 1D Decoder
            for ind, (dim_target, dim_source) in enumerate(reversed(in_out_1d)):
                is_last = ind >= (len(in_out_1d) - 1)

                if is_last:
                    up = nn.Conv1d(dim_source, dim_target, 3, padding=1)
                else:
                    up = Upsample1d(dim_source, dim_target)
                
                blocks = nn.ModuleList([])
                for _ in range(num_res_blocks):
                    blocks.append(ResnetBlock1d(dim_target, dim_target, time_emb_dim=None))
                    
                attn = LinearAttention1d(dim_target, heads=attn_heads, dim_head=attn_dim_head)
                
                self.ups_1d.append(nn.ModuleList([
                    up,
                    blocks,
                    attn
                ]))
                
        else:
            # Middle (Bottleneck) 2D
            self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=None)
            self.mid_attn = LinearAttention(mid_dim, heads=attn_heads, dim_head=attn_dim_head)
            self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=None)
            
            # To Z
            self.norm_out = RMSNorm(mid_dim, normalize_dim=1)
            self.to_z = nn.Conv2d(mid_dim, 2 * z_channels, 1)
            # # Init to_z to be the identity
            # nn.init.zeros_(self.to_z.weight)
            # nn.init.zeros_(self.to_z.bias)
            
            # --- Decoder ---
            # From Z
            self.from_z = nn.Conv2d(z_channels, mid_dim, 1)
            
            # Middle Decoder
            self.mid_dec_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=None)
            self.mid_dec_attn = LinearAttention(mid_dim, heads=attn_heads, dim_head=attn_dim_head)
            self.mid_dec_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=None)
        
        self.ups = nn.ModuleList([])
        
        # Reverse in_out for decoder
        # in_out: [(ch, ch*m1), (ch*m1, ch*m2), ...]
        # reversed: [(ch*m1, ch*m2), (ch, ch*m1)] (if len=2)
        # We want to iterate backwards.
        
        for ind, (dim_target, dim_source) in enumerate(reversed(in_out)):
            # dim_source is the larger channel count (input to decoder layer)
            # dim_target is the smaller channel count (output of decoder layer)
            
            # Encoder structure:
            # 0: Down (ch -> ch*m1)
            # 1: Conv (ch*m1 -> ch*m2) (if last)
            
            # Decoder structure (reversed):
            # 0: Conv (ch*m2 -> ch*m1) (if first)
            # 1: Up (ch*m1 -> ch)
            
            if ind == 0:
                up = nn.Conv2d(dim_source, dim_target, 3, padding=1)
            else:
                up = Upsample(dim_source, dim_target)
                
            blocks = nn.ModuleList([])
            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(dim_target, dim_target, time_emb_dim=None))
                
            attn = LinearAttention(dim_target, heads=attn_heads, dim_head=attn_dim_head)
            
            self.ups.append(nn.ModuleList([
                up,
                blocks,
                attn
            ]))
            
        self.norm_final = RMSNorm(ch, normalize_dim=1)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)
        
    def encode(self, x):
        h = self.conv_in(x)
        
        for blocks, attn, down in self.downs:
            for block in blocks:
                h = block(h)
            h = attn(h)
            h = down(h)
        
        if self.ch_mult_1d is not None:
            h = self.to_1d(h)
            
            for blocks, attn, down in self.downs_1d:
                for block in blocks:
                    h = block(h)
                h = attn(h)
                h = down(h)
            
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        
        h = self.norm_out(h)
        h = self.to_z(h)
        
        mean, logvar = torch.chunk(h, 2, dim=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        h = self.from_z(z)
        
        h = self.mid_dec_block1(h)
        h = self.mid_dec_attn(h)
        h = self.mid_dec_block2(h)
        
        if self.ch_mult_1d is not None:
            for up, blocks, attn in self.ups_1d:
                h = up(h)
                for block in blocks:
                    h = block(h)
                h = attn(h)
            
            h = self.from_1d(h)
        
        for up, blocks, attn in self.ups:
            h = up(h)
            for block in blocks:
                h = block(h)
            h = attn(h)
            
        h = self.norm_final(h)
        h = self.conv_out(h)
        return h

    def forward(self, x, sample_posterior=True, return_loss=False):
        mean, logvar = self.encode(x)
        if sample_posterior:
            z = self.reparameterize(mean, logvar)
        else:
            z = mean
        dec = self.decode(z)
        
        if self.training or return_loss:
            if self.binary_mode:
                target = (x + 1) * 0.5
                if self.loss_type == 'focal':
                    recon_loss = sigmoid_focal_loss(dec, target, gamma=self.focal_gamma, alpha=self.focal_alpha, reduction='mean')
                else:
                    recon_loss = F.binary_cross_entropy_with_logits(dec, target, reduction='mean')
            else:
                recon_loss = F.mse_loss(dec, x, reduction='mean')
            
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            kl_loss = kl_loss / x.shape[0]
            
            return dec, recon_loss, kl_loss
        
        return dec