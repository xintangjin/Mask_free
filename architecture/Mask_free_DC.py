import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import warnings
from torch import einsum
from einops import rearrange, repeat
import scipy.io as scio



def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_rgb = nn.LayerNorm(3)

    def forward(self, x, rgb=None, *args, **kwargs):
        x = self.norm(x)
        if rgb is not None:
            rgb = self.norm_rgb(rgb)
            return self.fn(x,rgb, *args, **kwargs)
        else:
            return self.fn(x, *args, **kwargs)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class HS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            window_size=(8, 8),
            dim_head=28,
            heads=8,
            only_local_branch=False,
            spectral_banch=True
    ):
        super().__init__()
        index = int(math.log2(heads))
        self.index = index
        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.only_local_branch = only_local_branch
        self.spectral_banch = spectral_banch
        h, w = 256 // self.heads, 320 // self.heads
        # position embedding
        if only_local_branch:
            seq_l = window_size[0] * window_size[1]
            self.pos_emb = nn.Parameter(torch.Tensor(1, heads, seq_l, seq_l))
            trunc_normal_(self.pos_emb)
        else:
            seq_l1 = window_size[0] * window_size[1]
            self.pos_emb1 = nn.Parameter(torch.Tensor(1, 1, heads//2, seq_l1, seq_l1))
            seq_l2 = h*w//seq_l1
            self.pos_emb2 = nn.Parameter(torch.Tensor(1, 1, heads//2, seq_l2, seq_l2))
            trunc_normal_(self.pos_emb1)
            trunc_normal_(self.pos_emb2)

        inner_dim = dim_head * heads
        self.num_heads = heads

        self.to_q1 = nn.Linear(dim +3* heads, inner_dim+3* heads, bias=False)
        self.to_kv1 = nn.Linear(dim +3* heads , (inner_dim+3* heads) * 2, bias=False)

        self.to_q = nn.Linear(dim +3* heads, inner_dim+3* heads, bias=False)
        self.to_kv = nn.Linear(dim +3 * heads, (inner_dim+3* heads) * 2, bias=False)
        self.to_out = nn.Linear(inner_dim+3* heads, dim)

        self.q_down = nn.Linear(int((2**(3-index))**2), 1)
        self.k_down = nn.Linear(int((2**(3-index))**2), 1)

        # self.q_down = nn.AvgPool2d(int(2**(3-index)),int(2**(3-index)))
        # self.k_down = nn.AvgPool2d(int(2**(3-index)),int(2**(3-index)))
        if index !=0:
            self.ref = nn.Conv2d(3, 6*index-3, 1, padding=0, bias=True)
        self.rgb_down = nn.AvgPool2d(int(2 ** index), int(2 ** index))

    def forward(self, x, rgb):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x.shape
        w_size = self.window_size
        assert h % w_size[0] == 0 and w % w_size[1] == 0, 'fmap dimensions must be divisible by the window size'
        rgb = rearrange(rgb, 'b h  w c -> b c h w')
        if self.index != 0:
            rgb = torch.cat([self.ref(rgb), rgb], dim=1)
        rgb = self.rgb_down(rgb)
        rgb = rearrange(rgb, 'b c h w -> b h  w c')
        if self.spectral_banch:
            x_inp = torch.cat([x, rgb], dim=-1)
            _, _, _, c1 = x_inp.shape
            x_in = rearrange(x_inp, 'b (h b0) (w b1) c -> (b h w) (b0 b1) c', h=32, w=40)
            q_inp = self.to_q1(x_in)
            k_inp, v_inp = self.to_kv1(x_in).chunk(2, dim=-1)
            q, k = map(lambda t: rearrange(t, 'b mm d -> b d mm'), (q_inp, k_inp))
            q, k = self.q_down(q), self.k_down(k)
            q, k = map(lambda t: rearrange(t, '(b h_in w_in) (h d) n -> b h (h_in w_in n) d', h=self.num_heads, h_in=32, w_in=40), (q, k))
            v = rearrange(v_inp, '(b h_in w_in) n (h d) -> b h (h_in w_in n) d', h=self.num_heads, h_in=32, w_in=40)

            # x_in = x.reshape(b, h * w, c)
            # q_inp = self.to_q(x_in)
            # k_inp, v_inp = self.to_kv(x_in).chunk(2, dim=-1)
            # q, k = map(lambda t: rearrange(t, 'b (h w) d -> b d h w', h=h, w=w), (q_inp, k_inp))
            # q, k = self.q_down(q), self.k_down(k)
            # q, k = map(lambda t: rearrange(t, 'b (h d) h_in w_in -> b h (h_in w_in) d', h=self.num_heads), (q, k))
            # v = rearrange(v_inp, 'b n (h d) -> b h n d', h=self.num_heads)

            q = q.transpose(-2, -1)
            k = k.transpose(-2, -1)
            v = v.transpose(-2, -1)

            q = F.normalize(q, dim=-1, p=2)
            k = F.normalize(k, dim=-1, p=2)
            attn = (k @ q.transpose(-2, -1))  # A = K^T*Q
            attn = attn * self.scale
            attn = attn.softmax(dim=-1)
            x_in = attn @ v  # b,heads,d,hw
            x_in = x_in.permute(0, 3, 1, 2)  # Transpose
            x = x_in.view(b, h, w, c1)[:,:,:,:c] + x

        if self.only_local_branch:
            x_inp = torch.cat([x, rgb], dim=-1)
            x_inp = rearrange(x_inp, 'b (h b0) (w b1) c -> (b h w) (b0 b1) c', b0=w_size[0], b1=w_size[1])
            q = self.to_q(x_inp)
            k, v = self.to_kv(x_inp).chunk(2, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
            q *= self.scale
            sim = einsum('b h i d, b h j d -> b h i j', q, k)
            sim = sim + self.pos_emb
            attn = sim.softmax(dim=-1)
            out = einsum('b h i j, b h j d -> b h i d', attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            out = self.to_out(out)
            out = rearrange(out, '(b h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1],
                            b0=w_size[0])

        else:
            x_inp = torch.cat([x, rgb], dim=-1)
            q = self.to_q(x_inp)
            k, v = self.to_kv(x_inp).chunk(2, dim=-1)
            q1, q2 = q[:,:,:,:c//2], q[:,:,:,c//2:]
            k1, k2 = k[:,:,:,:c//2], k[:,:,:,c//2:]
            v1, v2 = v[:,:,:,:c//2], v[:,:,:,c//2:]

            # local branch
            q1, k1, v1 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                              b0=w_size[0], b1=w_size[1]), (q1, k1, v1))
            q1, k1, v1 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads//2), (q1, k1, v1))
            q1 *= self.scale
            sim1 = einsum('b n h i d, b n h j d -> b n h i j', q1, k1)
            sim1 = sim1 + self.pos_emb1
            attn1 = sim1.softmax(dim=-1)
            out1 = einsum('b n h i j, b n h j d -> b n h i d', attn1, v1)
            out1 = rearrange(out1, 'b n h mm d -> b n mm (h d)')

            # non-local branch
            q2, k2, v2 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                                 b0=w_size[0], b1=w_size[1]), (q2, k2, v2))
            q2, k2, v2 = map(lambda t: t.permute(0, 2, 1, 3), (q2.clone(), k2.clone(), v2.clone()))
            q2, k2, v2 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads//2), (q2, k2, v2))
            q2 *= self.scale
            sim2 = einsum('b n h i d, b n h j d -> b n h i j', q2, k2)
            sim2 = sim2 + self.pos_emb2
            attn2 = sim2.softmax(dim=-1)
            out2 = einsum('b n h i j, b n h j d -> b n h i d', attn2, v2)
            out2 = rearrange(out2, 'b n h mm d -> b n mm (h d)')
            out2 = out2.permute(0, 2, 1, 3)

            out = torch.cat([out1,out2],dim=-1).contiguous()
            out = self.to_out(out)
            out = rearrange(out, 'b (h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1],
                            b0=w_size[0])
        return out[:,:,:,:c]

class HPAB(nn.Module):
    def __init__(
            self,
            dim,
            window_size=(8, 8),
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, HS_MSA(dim=dim, window_size=window_size, dim_head=dim_head, heads=heads, only_local_branch=(heads==1))),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, rgb=None):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        rgb = rgb.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, rgb) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class HPT(nn.Module):
    def __init__(self, in_dim=28, out_dim=28, dim=28, num_blocks=[1,1,1]):
        super(HPT, self).__init__()
        self.dim = dim
        self.scales = len(num_blocks)

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_scale = dim
        for i in range(self.scales-1):
            self.encoder_layers.append(nn.ModuleList([
                HPAB(dim=dim_scale, num_blocks=num_blocks[i], dim_head=dim, heads=dim_scale // dim),
                nn.Conv2d(dim_scale, dim_scale * 2, 4, 2, 1, bias=False)
            ]))
            dim_scale *= 2

        # Bottleneck
        self.bottleneck = HPAB(dim=dim_scale, dim_head=dim, heads=dim_scale // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(self.scales-1):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_scale, dim_scale // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_scale, dim_scale // 2, 1, 1, bias=False),
                HPAB(dim=dim_scale // 2, num_blocks=num_blocks[self.scales - 2 - i], dim_head=dim,
                     heads=(dim_scale // 2) // dim),
            ]))
            dim_scale //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        #### activation function
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, rgb=None):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        b, c, h_inp, w_inp = x.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        rgb= F.pad(rgb, [0, pad_w, 0, pad_h], mode='reflect')
        # Embedding
        fea = self.embedding(x)
        x = x[:,:self.dim,:,:]

        # Encoder
        fea_encoder = []
        for (HSAB, FeaDownSample) in self.encoder_layers:
            fea = HSAB(fea, rgb)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea, rgb)

        # Decoder
        for i, (FeaUpSample, Fution, HSAB) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.scales-2-i]], dim=1))
            fea = HSAB(fea, rgb)

        # Mapping
        out = self.mapping(fea) + x
        return out[:, :, :h_inp, :w_inp]

def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,1)
    return y

def At(y,Phi):
    temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    x = temp*Phi
    return x

def shift_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=step*i, dims=2)
    return inputs

def shift_back_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*step*i, dims=2)
    return inputs

class rgb_init(nn.Module):
    def __init__(self, dim=28, channel=128):
        super(rgb_init, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
                nn.Conv2d(dim*3, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, dim, 1, padding=0, bias=True),
                nn.Softplus())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, y, step=2):
        b, c, h, w = y.shape
        rgb3d = repeat(y, 'b c h w -> b c nc h w', nc=self.dim)
        rgb3d = rearrange(rgb3d, 'b c nc h w -> (b c) nc h w')
        # rgb3d = F.pad(rgb3d, pad=(0, (self.dim - 1) * step, 0, 0), mode='constant')
        for i in range(self.dim):
            rgb3d[:, i, :, :] = torch.roll(rgb3d[:, i, :, :], shifts=step * i, dims=2)
        rgb3d = rearrange(rgb3d, '(b c) nc h w  -> b (c nc) h w', b=b, c=c)
        rgb3d = self.mlp(rgb3d)
        rgb3d = self.relu(rgb3d)
        return rgb3d

class NoisePaNet(nn.Module):
    def __init__(self, in_nc=31, out_nc=3, channel=64):
        super(NoisePaNet, self).__init__()
        self.fution = nn.Conv2d(in_nc, channel, 1, 1, 0, bias=True)
        self.down_sample = nn.Conv2d(channel, channel, 3, (3 - 1) // 2, 1, bias=True)
        self.mlp = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
                nn.Softplus())
        self.relu = nn.ReLU(inplace=True)

        self.fution1 = nn.Conv2d(3, channel, 1, 1, 0, bias=True)

        self.down_sample1 = nn.Conv2d(channel, channel, 3, (3 - 1) // 2, 1, bias=True)
        self.mlp1 = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
                nn.Softplus())
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x_mask, x_rgb):

        x = self.down_sample(self.relu(self.fution(x_mask)))
        x = self.mlp(x)

        x_rgb = self.down_sample1(self.relu1(self.fution1(x_rgb)))
        x_rgb = self.mlp1(x_rgb)

        x = x - x_rgb + 1e-6
        return x

class HyPaNet(nn.Module):
    def __init__(self, in_nc=29, out_nc=8, channel=64):
        super(HyPaNet, self).__init__()
        self.fution = nn.Conv2d(in_nc, channel, 1, 1, 0, bias=True)
        self.down_sample = nn.Conv2d(channel, channel, 3, 2, 1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
                nn.Softplus())
        self.relu = nn.ReLU(inplace=True)
        self.out_nc = out_nc

    def forward(self, x):
        x = self.down_sample(self.relu(self.fution(x)))
        x = self.avg_pool(x)
        x = self.mlp(x) + 1e-6
        return x[:,:self.out_nc//2,:,:], x[:,self.out_nc//2:,:,:]

class spectral2rgb(nn.Module):
    def __init__(self, in_nc=28, channel=64, out_nc=3):
        super(spectral2rgb, self).__init__()
        self.mlp = nn.Sequential(
                nn.Conv2d(in_nc, channel, 1, 1, 0, bias=True),
                nn.Conv2d(channel, channel, 1,1, padding=0, bias=True),
                nn.Conv2d(channel, out_nc, 1,1, padding=0, bias=True))
    def forward(self, x, step=2, dim=28):
        [bs, nC, row, col] = x.shape
        x = self.mlp(x)
        inputs = torch.zeros((bs, 3, row, col)).cuda().float()
        inputs[:, :, :, 0:col- (dim - 1) * step] = x[:, :, :, 0:col- (dim - 1) * step]
        return inputs

class Mask_free_DC(nn.Module):
    def __init__(self, dim=28, num_iterations=1):
        super(Mask_free_DC, self).__init__()
        self.dim=dim
        self.para_estimator = HyPaNet(in_nc=dim, out_nc=num_iterations * 2)
        self.fution = nn.Conv2d(dim*2, dim, 1, padding=0, bias=True)
        self.fution_cassi_init = nn.Conv2d(dim*2, dim, 1, padding=0, bias=True)
        self.initial_rgb = nn.Sequential(
                nn.Conv2d(dim*3, dim, 1, padding=0, bias=True),
                nn.Conv2d(dim, dim*2, 1, padding=0, bias=True),
                nn.Conv2d(dim*2, dim, 1, padding=0, bias=True))
        self.noise_estimator = NoisePaNet(in_nc=dim, out_nc=3)
        self.spectral2rgb = spectral2rgb(in_nc=dim)
        self.z_init = nn.Conv2d(dim*2, dim, 1, padding=0, bias=True)

        self.num_iterations = num_iterations
        self.denoisers = nn.ModuleList([])
        for _ in range(num_iterations):
            self.denoisers.append(
                HPT(in_dim=dim+1+3, out_dim=dim, dim=dim, num_blocks=[1,1,1]),
            )

        self.fusion = nn.Conv2d(dim*2, dim, 1, padding=0, bias=False)
    def initial_cassi(self, y, Phi):
        """
        :param y: [b,256,310]
        :param Phi: [b,28,256,310]
        :return: temp: [b,28,256,310]; alpha: [b, num_iterations]; beta: [b, num_iterations]
        """
        nC, step = self.dim, 2
        y = y / nC * 2
        bs,row,col = y.shape
        y_shift = torch.zeros(bs, nC, row, col).cuda().float()
        for i in range(nC):
            y_shift[:, i, :, step * i:step * i + col - (nC - 1) * step] = y[:, :, step * i:step * i + col - (nC - 1) * step]
        z = self.fution_cassi_init(torch.cat([y_shift, Phi], dim=1))
        alpha, beta = self.para_estimator(self.fution(torch.cat([y_shift, Phi], dim=1)))
        return z, alpha, beta

    def forward(self, y, input_mask=None, rgb=None, rgb_s=None):
        """
        :param y: [b,256,310]
        :param Phi: [b,28,256,310]
        :param Phi_PhiT: [b,256,310]
        :return: z_crop: [b,28,256,256]
        """
        Phi, Phi_s = input_mask
        z, alphas, betas = self.initial_cassi(y, Phi)
        z_rgb = self.initial_rgb(rgb_s)
        noise = self.noise_estimator(shift_back_3d(z), rgb)
        z = self.z_init(torch.cat([z, z_rgb], dim=1))

        for i in range(self.num_iterations):
            #cassi
            alpha, beta = alphas[:, i, :, :], betas[:, i:i + 1, :, :]
            Phi_z = A(z, Phi)
            x = z + At(torch.div(y - Phi_z, alpha + Phi_s), Phi)
            x = shift_back_3d(x)

            #fusion rgb
            rgb_pre = self.spectral2rgb(x, dim=self.dim)
            rgb_div = rgb - noise - rgb_pre
            beta_repeat = beta.repeat(1, 1, x.shape[2], x.shape[3])

            z = self.denoisers[i](torch.cat([x, beta_repeat, rgb_div],dim=1), rgb)

            if i<self.num_iterations-1:
                z = shift_3d(z)
        return z[:, :, :, 0:256], rgb_pre[:, :, :, 0:256]


# if __name__=="__main__":
#     from thop import profile
#     dim=31
#     model = Mask_free_DC(dim=dim, num_iterations=3).cuda()
#     input = torch.randn(1,256, 256+(dim -1)*2).cuda()
#     Phi_batch = torch.randn(1, dim, 256, 256+(dim -1)*2).cuda()
#     Phi_s_batch = torch.randn(1, 256, 256+(dim -1)*2).cuda()
#     rgb = torch.randn(1, 3, 256, 256+(dim -1)*2).cuda()
#     rgb_shift = torch.randn(1, dim*3, 256, 256+(dim -1)*2).cuda()
#     input_mask = (Phi_batch, Phi_s_batch)
#     # z = model(input, rgb, input_mask)
#     # print(z.shape)
#     total_ops, total_params = profile(model, inputs=(input, input_mask, rgb, rgb_shift))
#     print(
#             "%s | %.2f | %.2f" % ("HPT", total_params / (1000 ** 2), total_ops / (1000 ** 3))
#         )