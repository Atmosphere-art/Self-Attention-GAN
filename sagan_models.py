import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle.nn.utils import spectral_norm
from paddle.nn.initializer import XavierUniform


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2D:
        XavierUniform((m.weight))
        m.bias[:] = 0.


def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups))


def snlinear(in_features, out_features):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features))


def sn_embedding(num_embeddings, embedding_dim):
    return spectral_norm(nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim))


class Self_Attn(paddle.nn.Layer):
    """ Self attention Layer"""

    def __init__(self, in_channels):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2D(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(axis=-1)
        # self.sigma = nn.Parameter(paddle.zeros(1))
        self.sigma = paddle.static.create_parameter(shape=[1], dtype='float32')

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        _ = x.shape[0]
        ch = x.shape[1]
        h = x.shape[2]
        w = x.shape[3]
        # _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.reshape([-1, ch//8, h*w])
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.reshape([-1, ch//8, h*w//4])
        # Attn map
        # attn = paddle.bmm(theta.permute(0, 2, 1), phi)
        attn = paddle.bmm(paddle.transpose(theta, perm=[0,2,1]), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.reshape([-1, ch//2, h*w//4])
        # Attn_g
        # attn_g = paddle.bmm(g, attn.permute(0, 2, 1))
        attn_g = paddle.bmm(g, paddle.transpose(attn, perm=[0,2,1]))
        attn_g = attn_g.reshape([-1, ch//2, h, w])
        attn_g = self.snconv1x1_attn(attn_g)
        # Out
        out = x + self.sigma*attn_g
        return out


class ConditionalBatchNorm2d(paddle.nn.Layer):
    # https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2D(num_features, momentum=0.001)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        # self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight[:, :num_features] = 1.  # Initialize scale to 1
        self.embed.weight[:, num_features:] = 0  # Initialize bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.reshape([-1, self.num_features, 1, 1]) * out + beta.reshape([-1, self.num_features, 1, 1])
        return out


class GenBlock(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, num_classes):
        super(GenBlock, self).__init__()
        self.cond_bn1 = ConditionalBatchNorm2d(in_channels, num_classes)
        self.relu = nn.ReLU()
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.cond_bn2 = ConditionalBatchNorm2d(out_channels, num_classes)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, labels):
        x0 = x

        x = self.cond_bn1(x, labels)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest') # upsample
        x = self.snconv2d1(x)
        x = self.cond_bn2(x, labels)
        x = self.relu(x)
        x = self.snconv2d2(x)

        x0 = F.interpolate(x0, scale_factor=2, mode='nearest') # upsample
        x0 = self.snconv2d0(x0)

        out = x + x0
        return out


class Generator(paddle.nn.Layer):
    """Generator."""

    def __init__(self, z_dim, g_conv_dim, num_classes):
        super(Generator, self).__init__()

        self.z_dim = z_dim
        self.g_conv_dim = g_conv_dim
        self.snlinear0 = snlinear(in_features=z_dim, out_features=g_conv_dim*16*4*4)
        self.block1 = GenBlock(g_conv_dim*16, g_conv_dim*16, num_classes)
        self.block2 = GenBlock(g_conv_dim*16, g_conv_dim*8, num_classes)
        self.block3 = GenBlock(g_conv_dim*8, g_conv_dim*4, num_classes)
        self.self_attn = Self_Attn(g_conv_dim*4)
        self.block4 = GenBlock(g_conv_dim*4, g_conv_dim*2, num_classes)
        self.block5 = GenBlock(g_conv_dim*2, g_conv_dim, num_classes)
        self.bn = nn.BatchNorm2D(g_conv_dim, epsilon=1e-5, momentum=0.0001)
        self.relu = nn.ReLU()
        self.snconv2d1 = snconv2d(in_channels=g_conv_dim, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        # Weight init
        self.apply(init_weights)

    def forward(self, z, labels):
        # n x z_dim
        # print("type(z): ", type(z))
        # print("z.shape", z.shape)
        # print("type(labels): ", type(labels))
        # print("labels.shape", labels.shape)
        act0 = self.snlinear0(z)            # n x g_conv_dim*16*4*4
        act0 = act0.reshape([-1, self.g_conv_dim*16, 4, 4]) # n x g_conv_dim*16 x 4 x 4
        act1 = self.block1(act0, labels)    # n x g_conv_dim*16 x 8 x 8
        act2 = self.block2(act1, labels)    # n x g_conv_dim*8 x 16 x 16
        act3 = self.block3(act2, labels)    # n x g_conv_dim*4 x 32 x 32
        act3 = self.self_attn(act3)         # n x g_conv_dim*4 x 32 x 32
        act4 = self.block4(act3, labels)    # n x g_conv_dim*2 x 64 x 64
        act5 = self.block5(act4, labels)    # n x g_conv_dim  x 128 x 128
        act5 = self.bn(act5)                # n x g_conv_dim  x 128 x 128
        act5 = self.relu(act5)              # n x g_conv_dim  x 128 x 128
        act6 = self.snconv2d1(act5)         # n x 3 x 128 x 128
        act6 = self.tanh(act6)              # n x 3 x 128 x 128
        return act6


class DiscOptBlock(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(DiscOptBlock, self).__init__()
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.AvgPool2D(2)
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x0 = x

        x = self.snconv2d1(x)
        x = self.relu(x)
        x = self.snconv2d2(x)
        x = self.downsample(x)

        x0 = self.downsample(x0)
        x0 = self.snconv2d0(x0)

        out = x + x0
        return out


class DiscBlock(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(DiscBlock, self).__init__()
        self.relu = nn.ReLU()
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.AvgPool2D(2)
        self.ch_mismatch = False
        if in_channels != out_channels:
            self.ch_mismatch = True
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, downsample=True):
        x0 = x

        x = self.relu(x)
        x = self.snconv2d1(x)
        x = self.relu(x)
        x = self.snconv2d2(x)
        if downsample:
            x = self.downsample(x)

        if downsample or self.ch_mismatch:
            x0 = self.snconv2d0(x0)
            if downsample:
                x0 = self.downsample(x0)

        out = x + x0
        return out


class Discriminator(paddle.nn.Layer):
    """Discriminator."""

    def __init__(self, d_conv_dim, num_classes):
        super(Discriminator, self).__init__()
        self.d_conv_dim = d_conv_dim
        self.opt_block1 = DiscOptBlock(3, d_conv_dim)
        self.block1 = DiscBlock(d_conv_dim, d_conv_dim*2)
        self.self_attn = Self_Attn(d_conv_dim*2)
        self.block2 = DiscBlock(d_conv_dim*2, d_conv_dim*4)
        self.block3 = DiscBlock(d_conv_dim*4, d_conv_dim*8)
        self.block4 = DiscBlock(d_conv_dim*8, d_conv_dim*16)
        self.block5 = DiscBlock(d_conv_dim*16, d_conv_dim*16)
        self.relu = nn.ReLU()
        self.snlinear1 = snlinear(in_features=d_conv_dim*16, out_features=1)
        self.sn_embedding1 = sn_embedding(num_classes, d_conv_dim*16)

        # Weight init
        self.apply(init_weights)
        XavierUniform(self.sn_embedding1.weight)

    def forward(self, x, labels):
        # n x 3 x 128 x 128
        h0 = self.opt_block1(x) # n x d_conv_dim   x 64 x 64
        h1 = self.block1(h0)    # n x d_conv_dim*2 x 32 x 32
        h1 = self.self_attn(h1) # n x d_conv_dim*2 x 32 x 32
        h2 = self.block2(h1)    # n x d_conv_dim*4 x 16 x 16
        h3 = self.block3(h2)    # n x d_conv_dim*8 x  8 x  8
        h4 = self.block4(h3)    # n x d_conv_dim*16 x 4 x  4
        h5 = self.block5(h4, downsample=False)  # n x d_conv_dim*16 x 4 x 4
        h5 = self.relu(h5)              # n x d_conv_dim*16 x 4 x 4
        h6 = paddle.sum(h5, axis=[2,3])   # n x d_conv_dim*16
        output1 = paddle.squeeze(self.snlinear1(h6)) # n
        # Projection
        h_labels = self.sn_embedding1(labels)   # n x d_conv_dim*16
        proj = paddle.multiply(h6, h_labels)          # n x d_conv_dim*16
        output2 = paddle.sum(proj, axis=[1])      # n
        # Out
        output = output1 + output2              # n
        return output
