import math
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
from einops import rearrange
from .utils import pad_tensor, pad_tensor_back



class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out


class AdaptiveAttentionEnhancement(nn.Module):
    def __init__(self, in_channels, channels, aae_variant='full'):
        super(AdaptiveAttentionEnhancement, self).__init__()
        if aae_variant not in ('full', 'no_norm', 'fixed_fusion'):
            raise ValueError(f'Unsupported AAE variant: {aae_variant}')
        self.aae_variant = aae_variant
        self.conv1_1 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2_1 = nn.Conv2d(channels // 2, channels // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_2 = nn.Conv2d(channels // 2, channels // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.instance = nn.InstanceNorm2d(channels // 2, affine=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(channels, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv1_2 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv2d(channels, channels, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=7 // 2, bias=False), nn.Sigmoid()
        )
        if aae_variant == 'fixed_fusion':
            self.fusion_gain = nn.Parameter(torch.ones(1))
            self.fusion_bias = nn.Parameter(torch.zeros(1))
        else:
            self.alpha = nn.Parameter(torch.ones(1))
            self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x1 = self.conv1_1(x)
        x1, x2 = torch.chunk(x1, 2, dim=1)
        if self.aae_variant == 'no_norm':
            out_instance = x1 * self.instance.weight[None, :, None, None]
            out_instance = out_instance + self.instance.bias[None, :, None, None]
        else:
            out_instance = self.instance(x1)
        out_identity = x2
        out1 = self.conv2_1(out_instance)
        out2 = self.conv2_2(out_identity)
        xp = torch.cat((out1, out2), 1)

        x = self.avg_pool(xp)
        x1 = self.conv1_2(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)
        x2 = self.fc(x).squeeze(-1).transpose(-1, -2)
        out1 = torch.sum(torch.matmul(x1, x2), dim=1).unsqueeze(-1).unsqueeze(-1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2), x1.transpose(-1, -2)), dim=1).unsqueeze(-1).unsqueeze(-1)
        out2 = self.sigmoid(out2)
        out = self.mix(out1, out2)
        out = self.conv1_2(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        ca_out = xp * out

        avg_out = torch.mean(xp, dim=1, keepdim=True)
        max_out, _ = torch.max(xp, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        spatial_weight = self.spatial_att(concat)
        sa_out = spatial_weight * xp

        if self.aae_variant == 'fixed_fusion':
            channel_ratio = math.e / (math.e + 1.0)
            fused_attn = self.fusion_gain * (
                channel_ratio * ca_out + (1.0 - channel_ratio) * sa_out
            ) + self.fusion_bias
        else:
            weights = torch.softmax(torch.stack([self.alpha, self.beta]), dim=0)
            fused_attn = weights[0] * ca_out + weights[1] * sa_out

        return fused_attn


class SimpleNetwork(nn.Module):

    def __init__(self, dim=8, expand=2, aae_variant='full'):
        super().__init__()
        self.dim = dim
        self.stage = 2
        self.in_proj = nn.Conv2d(3, dim, 1, 1, 0, bias=False)
        self.enc = AdaptiveAttentionEnhancement(dim, dim, aae_variant=aae_variant)
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(2):
            self.encoder_layers.append(
                nn.ModuleList([
                    nn.Conv2d(dim_stage, dim_stage * expand, 1, 1, 0, bias=False),
                    nn.Conv2d(dim_stage * expand, dim_stage * expand, 3, 2, 1, bias=False, groups=dim_stage * expand),
                    nn.Conv2d(dim_stage * expand, dim_stage * expand, 1, 1, 0, bias=False),
                ]))
            dim_stage *= 2
        self.decoder_layers = nn.ModuleList([])
        for i in range(2):
            self.decoder_layers.append(
                nn.ModuleList([
                    nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                    nn.Conv2d(dim_stage // 2, dim_stage, 1, 1, 0, bias=False),
                    nn.Conv2d(dim_stage, dim_stage, 3, 1, 1, bias=False, groups=dim_stage),
                    nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, 0, bias=False),
                ]))
            dim_stage //= 2
        self.out_conv2 = nn.Conv2d(self.dim, 3, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        x, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(x, divide=16)
        fea = self.lrelu(self.in_proj(x))
        fea = self.enc(fea)
        fea_encoder = []
        for (Conv1, Conv2, Conv3) in self.encoder_layers:
            fea_encoder.append(fea)
            fea = Conv3(self.lrelu(Conv2(self.lrelu(Conv1(fea)))))

        fea_decoder = [fea]
        for i, (FeaUpSample, Conv1, Conv2, Conv3) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Conv3(self.lrelu(Conv2(self.lrelu(Conv1(fea)))))
            fea = fea + fea_encoder[self.stage - 1 - i]
            fea_decoder.append(fea)

        out = self.out_conv2(fea)
        out_feature = pad_tensor_back(fea, pad_left, pad_right, pad_top, pad_bottom)
        out = pad_tensor_back(out, pad_left, pad_right, pad_top, pad_bottom)
        return out, out_feature, fea_decoder


class MySampler(nn.Module):

    def __init__(self, sampler_input_resolution=256, sampler_output_resolution=256):
        super().__init__()
        self.n_vertices = sampler_output_resolution

        self.generator = LightBackbone(input_resolution=sampler_input_resolution, extra_pooling=True, n_base_feats=8)
        self.intervals_generator = nn.Linear(256, (self.n_vertices - 1) * 2)
        self.init_weights()

    def init_weights(self):
        def special_initilization(m):
            classname = m.__class__.__name__
            if 'Conv' in classname:
                nn.init.xavier_normal_(m.weight.data)
            elif 'InstanceNorm' in classname:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

        self.generator.apply(special_initilization)
        nn.init.zeros_(self.intervals_generator.weight)
        nn.init.ones_(self.intervals_generator.bias)

    def forward(self, img):
        code = self.generator(img)
        code = code.view(code.shape[0], -1)
        intervals = self.intervals_generator(code).view(code.shape[0], -1, self.n_vertices - 1)
        intervals = intervals.softmax(-1)

        vertieces = F.pad(intervals.cumsum(-1), (1, 0), 'constant', 0)
        grid_batch = []
        for b in range(code.shape[0]):
            xx, yy = torch.meshgrid(vertieces[b, 0, :], vertieces[b, 1, :])
            xx = xx.unsqueeze(0).unsqueeze(0)
            yy = yy.unsqueeze(0).unsqueeze(0)
            xx = xx * 2 - 1
            yy = yy * 2 - 1
            grid_batch.append(torch.cat([yy, xx], dim=1).permute(0, 2, 3, 1))
        self.vertieces = vertieces
        grid = torch.cat(grid_batch, dim=0)
        img_sampled = F.grid_sample(img, grid, mode='bilinear', align_corners=False, padding_mode='zeros')
        return img_sampled, grid


class BasicBlock(nn.Sequential):
    r"""The basic block module (Conv+LeakyReLU[+InstanceNorm]).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm=False):
        body = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1), nn.LeakyReLU(0.2)]
        if norm:
            body.append(nn.InstanceNorm2d(out_channels, affine=True))
        super(BasicBlock, self).__init__(*body)


class LightBackbone(nn.Sequential):
    r"""The 5-layer CNN backbone module in [TPAMI 3D-LUT]
        (https://github.com/HuiZeng/Image-Adaptive-3DLUT).
    Args:
        input_resolution (int, optional): Resolution for pre-downsampling. Default: 256.
        extra_pooling (bool, optional): Whether to use an extra pooling layer at the end
            of the backbone. Default: False.
        n_base_feats (int, optional): Channel multiplier. Default: 8.
    """

    def __init__(self, input_resolution=256, extra_pooling=True, n_base_feats=8, **kwargs) -> None:
        body = [BasicBlock(3, n_base_feats, stride=2, norm=True)]
        n_feats = n_base_feats
        for _ in range(3):
            body.append(BasicBlock(n_feats, n_feats * 2, stride=2, norm=True))
            n_feats = n_feats * 2
        body.append(BasicBlock(n_feats, n_feats, stride=2))
        body.append(nn.Dropout(p=0.5))
        if extra_pooling:
            body.append(nn.AdaptiveAvgPool2d(2))
        super().__init__(*body)
        self.input_resolution = input_resolution
        self.out_channels = n_feats * (4 if extra_pooling else (input_resolution // 32) ** 2)

    def forward(self, imgs):
        imgs = F.interpolate(imgs, size=(self.input_resolution,) * 2, mode='bilinear', align_corners=False)
        return super().forward(imgs).view(imgs.shape[0], -1)


class Res18Backbone(nn.Module):
    r"""The ResNet-18 backbone.
    Args:
        pretrained (bool, optional): Whether to use the torchvison pretrained weights.
            Default: True.
        input_resolution (int, optional): Resolution for pre-downsampling. Default: 224.
    """

    def __init__(self, pretrained=True, input_resolution=224, **kwargs):
        super().__init__()
        import torchvision
        net = torchvision.models.resnet18(pretrained=pretrained)
        net.fc = nn.Identity()
        self.net = net
        self.input_resolution = input_resolution
        self.out_channels = 512

    def forward(self, imgs):
        imgs = F.interpolate(imgs, size=(self.input_resolution,) * 2, mode='bilinear', align_corners=False)
        return self.net(imgs).view(imgs.shape[0], -1)


def lut_transform(imgs, luts):
    imgs = (imgs - .5) * 2.
    grids = imgs.permute(0, 2, 3, 1).unsqueeze(1)
    outs = F.grid_sample(luts, grids, mode='bilinear', padding_mode='border', align_corners=True)
    outs = outs.squeeze(2)
    return outs


class LUT3DGenerator(nn.Module):
    r"""The 3DLUT generator module.
    Args:
        n_colors (int): Number of input color channels.
        n_vertices (int): Number of sampling points along each lattice dimension.
        n_feats (int): Dimension of the input image representation vector.
        n_ranks (int): Number of ranks (or the number of basis LUTs).
    """

    def __init__(self, n_colors, n_vertices, n_feats, n_ranks) -> None:
        super().__init__()

        self.weights_generator = nn.Linear(n_feats, n_ranks)
        self.basis_luts_bank = nn.Linear(n_ranks, n_colors * (n_vertices ** n_colors), bias=False)

        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.n_feats = n_feats
        self.n_ranks = n_ranks

    def init_weights(self):
        r"""Init weights for models.
        For the mapping f (`backbone`) and h (`lut_generator`), we follow the initialization in
            [TPAMI 3D-LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).
        """
        nn.init.ones_(self.weights_generator.bias)
        identity_lut = torch.stack([
            torch.stack(torch.meshgrid(*[torch.arange(self.n_vertices)
                                         for _ in range(self.n_colors)]), dim=0).div(self.n_vertices - 1).flip(0),
            *[torch.zeros(self.n_colors, *((self.n_vertices,) * self.n_colors)) for _ in range(self.n_ranks - 1)]
        ], dim=0).view(self.n_ranks, -1)
        self.basis_luts_bank.weight.data.copy_(identity_lut.t())

    def forward(self, x):
        weights = self.weights_generator(x)
        luts = self.basis_luts_bank(weights)
        luts = luts.view(x.shape[0], -1, *((self.n_vertices,) * self.n_colors))
        return weights, luts

    def regularizations(self, smoothness, monotonicity):
        basis_luts = self.basis_luts_bank.weight.t().view(self.n_ranks, self.n_colors,
                                                          *((self.n_vertices,) * self.n_colors))
        tv, mn = 0, 0
        for i in range(2, basis_luts.ndimension()):
            diff = torch.diff(basis_luts.flip(i), dim=i)
            tv += torch.square(diff).sum(0).mean()
            mn += F.relu(diff).sum(0).mean()
        reg_smoothness = smoothness * tv
        reg_monotonicity = monotonicity * mn
        return reg_smoothness, reg_monotonicity


class SepLUT(nn.Module):

    def __init__(self, n_ranks=3, n_vertices_3d=17, n_vertices_1d=0, n_vertices_2d=0,
                 backbone='light',
                 input_resolution=256,
                 n_base_feats=8,
                 n_colors=3):
        super().__init__()
        assert backbone in ['light', 'res18']
        assert n_vertices_3d > 0

        self.backbone = dict(
            light=LightBackbone, res18=Res18Backbone)[backbone.lower()](
            input_resolution=input_resolution, extra_pooling=True, n_base_feats=n_base_feats)

        if n_vertices_3d > 0:
            self.lut3d_generator = LUT3DGenerator(n_colors, n_vertices_3d, self.backbone.out_channels, n_ranks)

        self.n_ranks = n_ranks
        self.n_colors = n_colors
        self.n_vertices_3d = n_vertices_3d
        self.n_vertices_1d = n_vertices_1d
        self.n_vertices_2d = n_vertices_2d
        self.backbone_name = backbone.lower()
        self.input_resolution = input_resolution

        self.init_weights()

    def init_weights(self):
        r"""Init weights for models.
        For the backbone network and the 3D LUT generator, we follow the initialization in
            [TPAMI 3D-LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).
        """

        def special_initilization(m):
            classname = m.__class__.__name__
            if 'Conv' in classname:
                nn.init.xavier_normal_(m.weight.data)
            elif 'InstanceNorm' in classname:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

        if self.backbone_name not in ['res18']:
            self.backbone.apply(special_initilization)
        if self.n_vertices_3d > 0:
            self.lut3d_generator.init_weights()

    def forward(self, imgs, origin_imgs):
        codes = self.backbone(imgs)
        if self.n_vertices_3d > 0:
            lut3d_weights, lut3d = self.lut3d_generator(codes)
            imgs = lut_transform(origin_imgs, lut3d)
        else:
            lut3d_weights = imgs.new_zeros(1)
        outs = imgs
        return outs


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):

    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):

    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):

    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):

    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        """
        Args:
            x: Tensor, shape (B, 8, H, W)
            y: Tensor, shape (B, 8, 256, 256)
        Returns:
            out: Tensor, shape (B, 8, H, W)
        """
        b, c, h, w = x.shape
        b, c, h2, w2 = y.shape

        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(y))

        k = F.adaptive_avg_pool2d(k, (h2, w2))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1)
        attn = attn * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 3, padding=1, bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim // reduction, dim, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class MultiScaleCrossAttentionFusion(nn.Module):

    def __init__(self, dim, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(MultiScaleCrossAttentionFusion, self).__init__()
        self.reduce_conv = nn.Conv2d(in_channels=32 + 16 + 8, out_channels=dim, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.channel_attn = ChannelAttention(dim)

    def forward(self, F_glo, F_loc_list):
        """
        Args:
            F_glo: Tensor, shape (B, 3, H, W)
            F_loc_list: list of 3 Tensors:
               - (B, 32, 64, 64)
               - (B, 16, 128, 128)
               - (B, 8, 256, 256)
        Returns:
            F_out: Tensor, shape (B, 8, H, W)
        """
        up_feats = []
        for f in F_loc_list:
            up = F.interpolate(f, size=(256, 256), mode='bilinear', align_corners=False)
            up_feats.append(up)
        F_loc_cat = torch.cat(up_feats, dim=1)
        F_loc = self.reduce_conv(F_loc_cat)

        F_loc = F.gelu(self.conv1(F_loc))
        F_glo = F.gelu(self.conv2(F_glo))
        F_glo = self.norm1(F_glo)
        F_loc = self.norm1(F_loc)

        F_glo = F_glo + self.attn(F_glo, F_loc)
        F_glo = self.channel_attn(F_glo)

        F_out = F_glo + self.ffn(self.norm2(F_glo))

        return F_out


class SeparableConv2d(nn.Module):
    """深度可分离卷积"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class AttentionGate(nn.Module):
    """SE-Style 通道注意力门控"""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MultiScaleDetailExtractor(nn.Module):
    """轻量化多尺度高频细节提取，保证上采样到原图精确尺寸"""

    def __init__(self, channels):
        super().__init__()
        self.pool1 = nn.AvgPool2d(2, stride=2)
        self.pool2 = nn.AvgPool2d(4, stride=4)

        self.conv1 = SeparableConv2d(channels, channels)
        self.conv2 = SeparableConv2d(channels, channels)
        self.conv3 = SeparableConv2d(channels, channels)

        self.fuse = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape

        x1 = self.conv1(x)

        y2 = self.conv2(self.pool1(x))
        x2 = F.interpolate(y2, size=(H, W), mode='bilinear', align_corners=False)

        y3 = self.conv3(self.pool2(x))
        x3 = F.interpolate(y3, size=(H, W), mode='bilinear', align_corners=False)

        out = self.fuse(torch.cat([x1, x2, x3], dim=1))
        return out


class DetailRecoveryModuleV2(nn.Module):
    """
    改进版细节恢复模块：
      - 多尺度细节提取
      - 通道注意力门控融合
      - 身份初始化（strength=0 时无干扰）
    """

    def __init__(self, channels):
        super().__init__()
        self.extractor = MultiScaleDetailExtractor(channels)
        self.attn_gate = AttentionGate(channels)
        self.strength = nn.Parameter(torch.zeros(1))
        self._init_identity()

    def _init_identity(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.constant_(m.weight, 0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        detail = self.extractor(x)
        gated = self.attn_gate(detail)
        return x + self.strength * gated


class MSCDNet(nn.Module):

    def __init__(self,
                 input_resolution=256,
                 sampler_input_resolution=256,
                 sampler_output_resolution=256,
                 n_vertices_3d=17,
                 n_vertices_2d=0,
                  n_vertices_1d=0,
                  aae_variant='full'):
        super().__init__()
        self.input_resolution = input_resolution
        self.autoencoder = SimpleNetwork(dim=8, expand=2, aae_variant=aae_variant)
        self.sampler = MySampler(
            sampler_input_resolution=sampler_input_resolution, sampler_output_resolution=sampler_output_resolution)
        self.lut = SepLUT(
            input_resolution=256, n_vertices_3d=n_vertices_3d, n_vertices_2d=n_vertices_2d, n_vertices_1d=n_vertices_1d)

        self.fusion = MultiScaleCrossAttentionFusion(dim=8, num_heads=2, ffn_expansion_factor=2)
        self.detail_recovery = DetailRecoveryModuleV2(channels=8)
        self.map = nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, img):
        img_unids = F.interpolate(img, size=(self.input_resolution,) * 2, mode='bilinear', align_corners=False)
        param_map, out_feature, loc_list = self.autoencoder(img_unids)
        img_input, grid = self.sampler(img=img)
        output_hr = self.lut(img_input, img)
        output = self.fusion(output_hr, loc_list)
        output = self.map(self.detail_recovery(output))
        return output


CoNet = MSCDNet
