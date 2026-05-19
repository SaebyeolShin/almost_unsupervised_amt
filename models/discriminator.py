import torch
import torch.nn as nn
import torch.nn.functional as F

def WNConv2d(*args, **kwargs):
    act = kwargs.pop("act", True)
    norm_layer = kwargs.pop("norm_layer", nn.utils.weight_norm)
    conv = norm_layer(nn.Conv2d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.2, True))

class DiscriminatorResBlock(nn.Module):
    def __init__(self, input_nc, ndf, kernel_size, stride, padding):
        super().__init__()
        self.block = WNConv2d(input_nc, ndf, kernel_size=kernel_size, stride=stride, padding=padding)
        
        # Shortcut
        # Check if we need projection
        do_proj = (input_nc != ndf) or (stride != 1 and stride != (1, 1))
        
        if do_proj:
            self.shortcut = WNConv2d(input_nc, ndf, kernel_size=1, stride=stride, padding=0, act=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.block(x) + self.shortcut(x)

class SpectrogramDiscriminator(nn.Module):
    """
    Discriminator with anisotropic kernels and weight normalization.
    Inspired by DAC's MRD but for 2D inputs directly.
    Now includes channel progression (doubling channels per layer up to max_ndf).
    """
    def __init__(self, input_nc=1, ndf=32, n_layers=4, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4), use_residual=False, max_ndf=128):
        super().__init__()

        layers = []
        # Initial layer
        layers.append(WNConv2d(input_nc, ndf, kernel_size=kernel_size, stride=stride, padding=padding))

        curr_ndf = ndf
        for i in range(n_layers - 1):
            # Double channels each layer, capped at max_ndf
            next_ndf = min(curr_ndf * 2, max_ndf)
            if use_residual:
                layers.append(DiscriminatorResBlock(curr_ndf, next_ndf, kernel_size=kernel_size, stride=stride, padding=padding))
            else:
                layers.append(WNConv2d(curr_ndf, next_ndf, kernel_size=kernel_size, stride=stride, padding=padding))
            curr_ndf = next_ndf

        # Post processing
        layers.append(WNConv2d(curr_ndf, curr_ndf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        layers.append(WNConv2d(curr_ndf, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), act=False))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        fmap = []
        for layer in self.layers:
            x = layer(x)
            fmap.append(x)
        return fmap

class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator.
    Downsamples input at each scale and runs a SpectrogramDiscriminator.
    """
    def __init__(self, input_nc=1, ndf=32, n_layers=4, scales=3, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4), use_residual=False):
        super().__init__()
        self.discriminators = nn.ModuleList()
        self.scales = scales
        
        for _ in range(scales):
            self.discriminators.append(
                SpectrogramDiscriminator(input_nc, ndf, n_layers, kernel_size, stride, padding, use_residual=use_residual)
            )
            
        self.downsample = nn.AvgPool2d(kernel_size=4, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        fmaps = []
        for i, d in enumerate(self.discriminators):
            if i > 0:
                x = self.downsample(x)
            fmaps.append(d(x))
        return fmaps

class GANLoss(nn.Module):
    """
    Computes LSGAN loss and Feature Matching loss.
    """
    def __init__(self, use_fm_loss=True, normalize_fm=True):
        super().__init__()
        self.use_fm_loss = use_fm_loss
        self.normalize_fm = normalize_fm

    def discriminator_loss(self, fmaps_real, fmaps_fake):
        """
        fmaps_real: List[List[Tensor]] (scales -> layers)
        fmaps_fake: List[List[Tensor]]
        """
        loss_d = 0
        loss_d_real = 0
        loss_d_fake = 0
        
        # Iterate over scales
        for scale_real, scale_fake in zip(fmaps_real, fmaps_fake):
            # The last layer output is the prediction score
            score_real = scale_real[-1]
            score_fake = scale_fake[-1]
            
            l_real = torch.mean((1 - score_real) ** 2)
            l_fake = torch.mean(score_fake ** 2)
            
            loss_d_real += l_real
            loss_d_fake += l_fake
            loss_d += l_real + l_fake
            
        # Calculate metrics (using the last scale as proxy or average across scales?)
        # Let's use the last scale (highest resolution) for metrics
        with torch.no_grad():
            # Last scale scores
            s_real = fmaps_real[-1][-1]
            s_fake = fmaps_fake[-1][-1]
            
            # Mean scores
            mean_score_real = torch.mean(s_real)
            mean_score_fake = torch.mean(s_fake)
            
            # Accuracy (threshold 0.5)
            # Real should be > 0.5
            acc_real = torch.mean((s_real > 0.5).float())
            # Fake should be < 0.5
            acc_fake = torch.mean((s_fake < 0.5).float())
            
        metrics = {
            "acc_real": acc_real,
            "acc_fake": acc_fake,
            "score_real": mean_score_real,
            "score_fake": mean_score_fake
        }
            
        return loss_d, loss_d_real, loss_d_fake, metrics

    def adversarial_loss(self, fmaps_fake):
        """
        Computes the adversarial loss for the generator (G wants D to predict 1).
        Likelihood that 'fake' is real.
        """
        loss_g = 0
        for scale_fake in fmaps_fake:
            score_fake = scale_fake[-1]
            loss_g += torch.mean((1 - score_fake) ** 2)
        return loss_g

    def feature_matching_loss(self, fmaps_real, fmaps_fake):
        """
        Computes Feature Matching loss between two sets of feature maps (e.g. Real and Rec).
        """
        if not self.use_fm_loss:
            return torch.tensor(0.0, device=fmaps_real[0][0].device)

        loss_fm = 0
        num_fm_terms = 0

        for scale_real, scale_fake in zip(fmaps_real, fmaps_fake):
            # Iterate over layers (excluding the last one which is the score)
            for feat_real, feat_fake in zip(scale_real[:-1], scale_fake[:-1]):
                loss_fm += torch.mean(torch.abs(feat_real.detach() - feat_fake))
                num_fm_terms += 1

        if self.normalize_fm and num_fm_terms > 0:
            loss_fm = loss_fm / num_fm_terms
            
        return loss_fm