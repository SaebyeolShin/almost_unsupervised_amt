import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange
from midi_autoencoder.lucidrains_ae import ResnetBlock, Downsample, Upsample, LinearAttention

def append_dims(t, dims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * dims))

class AsymmetricUpsample(nn.Module):
    def __init__(self, dim, dim_out=None, factor=(1, 2)):
        super().__init__()
        self.factor_h, self.factor_w = factor
        self.factor_product = self.factor_h * self.factor_w

        dim_out = dim_out if dim_out is not None else dim

        self.conv = nn.Conv2d(dim, dim_out * self.factor_product, 1)

        self.net = nn.Sequential(
            self.conv,
            nn.SiLU(),
            Rearrange('b (c h2 w2) h w -> b c (h h2) (w w2)', h2=self.factor_h, w2=self.factor_w)
        )

        self.init_conv_(self.conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // self.factor_product, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r=self.factor_product)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)

class CqtToLatentGenerator(nn.Module):
    """
    G_A: CQT (256x352) -> Latent (32x11)
    """
    def __init__(self, ngf=64, z_channels=4, aux_channels=0, ch_mult=[1, 2, 4, 8, 8], num_res_blocks=2, attn_heads=4, attn_dim_head=32, attn_start_level=0):
        super().__init__()

        self.ngf = ngf
        self.ch_mult = ch_mult
        self.strides = [(2, 2), (2, 2), (2, 2), (1, 2), (1, 2)]

        assert len(ch_mult) == len(self.strides), "ch_mult length must match strides length (5)"

        self.conv_in = nn.Conv2d(1, ngf * ch_mult[0], 3, padding=1)

        self.layers = nn.ModuleList()

        curr_ch = ngf * ch_mult[0]

        for i in range(len(ch_mult)):
            mult = ch_mult[i]
            mult_next = ch_mult[i+1] if i < len(ch_mult) - 1 else None
            stride = self.strides[i]

            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(curr_ch, curr_ch))

            if i >= attn_start_level:
                attn = LinearAttention(curr_ch, heads=attn_heads, dim_head=attn_dim_head)
            else:
                attn = nn.Identity()

            if i < len(ch_mult) - 1:
                out_ch = ngf * mult_next
                if stride == (2, 2):
                    down = Downsample(curr_ch, out_ch)
                else:
                    down = nn.Conv2d(curr_ch, out_ch, kernel_size=(3, 3), stride=stride, padding=(1, 1))
            else:
                out_ch = z_channels + aux_channels
                down = nn.Conv2d(curr_ch, out_ch, kernel_size=(3, 3), stride=stride, padding=(1, 1))

            self.layers.append(nn.ModuleList([blocks, attn, down]))
            curr_ch = out_ch

    def forward(self, x):
        x = self.conv_in(x)

        for blocks, attn, down in self.layers:
            for block in blocks:
                x = block(x)
            x = attn(x)
            x = down(x)

        return x

class LatentToCqtGenerator(nn.Module):
    """
    G_B: Latent (32x11) -> CQT (256x352)
    """
    def __init__(self, ngf=64, z_channels=4, aux_channels=0, ch_mult=[1, 2, 4, 8, 8], num_res_blocks=2, attn_heads=4, attn_dim_head=32, attn_start_level=0):
        super().__init__()

        self.ngf = ngf
        self.ch_mult = ch_mult
        self.strides = [(2, 2), (2, 2), (2, 2), (1, 2), (1, 2)]

        assert len(ch_mult) == len(self.strides), "ch_mult length must match strides length (5)"

        self.conv_in = nn.Conv2d(z_channels + aux_channels, ngf * ch_mult[-1], 3, padding=1)

        self.layers = nn.ModuleList()

        curr_ch = ngf * ch_mult[-1]

        for i in range(len(ch_mult) - 1, -1, -1):
            if i > 0:
                out_ch = ngf * ch_mult[i-1]
            else:
                out_ch = ngf * ch_mult[0]

            stride = self.strides[i]

            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(curr_ch, curr_ch))

            if i >= attn_start_level:
                attn = LinearAttention(curr_ch, heads=attn_heads, dim_head=attn_dim_head)
            else:
                attn = nn.Identity()

            if stride == (2, 2):
                up = Upsample(curr_ch, out_ch)
            else:
                up = AsymmetricUpsample(curr_ch, out_ch, factor=stride)

            self.layers.append(nn.ModuleList([blocks, attn, up]))
            curr_ch = out_ch

        self.block_out = ResnetBlock(curr_ch, curr_ch)
        self.conv_out = nn.Conv2d(curr_ch, 1, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)

        for blocks, attn, up in self.layers:
            for block in blocks:
                x = block(x)
            x = attn(x)
            x = up(x)

        x = self.block_out(x)
        x = self.conv_out(x)
        x = torch.tanh(x)
        return x
