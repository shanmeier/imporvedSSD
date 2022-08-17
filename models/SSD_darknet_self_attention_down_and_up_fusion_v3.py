import os
import torch
import torch.nn as nn
# from .criss_cross_attention import CrissCrossAttention


class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation=True):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        # return out, attention
        return out


class SA_down(nn.Module):
    def __init__(self):
        super(SA_down, self).__init__()
        self.attentions = nn.ModuleList([
            SelfAttention(256),
            SelfAttention(512),
            SelfAttention(1024),
            SelfAttention(256),
            SelfAttention(256),
            SelfAttention(256)
        ])
        self.down_features = nn.ModuleList([
            nn.Conv2d(256, 256, 3, 2, 1, bias=False),
            nn.Conv2d(512, 512, 3, 2, 1, bias=False),
            nn.Conv2d(1024, 256, 3, 2, 1, bias=False),
            nn.Conv2d(256, 256, 3, 1, 0, bias=False),
            nn.Conv2d(256, 256, 3, 1, 0, bias=False)
        ])
        self.down_channels = nn.ModuleList([
            nn.Conv2d(512, 256, 1, 1),
            nn.Conv2d(1024, 512, 1, 1)
        ])

    def forward(self, inputs):
        p3, p4, p5 = inputs
        p3 = self.attentions[0](p3)
        p4 = self.attentions[1](torch.cat([self.down_features[0](p3), self.down_channels[0](p4)], dim=1))
        p5 = self.attentions[2](torch.cat([self.down_features[1](p4), self.down_channels[1](p5)], dim=1))
        p6 = self.attentions[3](self.down_features[2](p5))
        p7 = self.attentions[4](self.down_features[3](p6))
        p8 = self.attentions[5](self.down_features[4](p7))

        return p3, p4, p5, p6, p7, p8


class DSF(nn.Module):
    def __init__(self):
        super(DSF, self).__init__()
        self.conv_p6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.gap_p6 = nn.AdaptiveAvgPool2d(1)
        self.gap_p7 = nn.AdaptiveAvgPool2d(1)
        self.sa = SelfAttention(256)

    def forward(self, deep_features):
        p6, p7, p8 = deep_features
        p6 = self.sa(self.conv_p6(p6) + self.gap_p6(p6) + self.gap_p7(p7) + p8)
        return p6


# 空间注意力模块
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # self.act=SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out)) * x
        return out



class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class Up_Fusion(nn.Module):
    def __init__(self):
        super(Up_Fusion, self).__init__()
        self.up_channels = nn.Conv2d(256, 1024, 3, 1, 1)
        self.down_channels = nn.ModuleList([
            nn.Conv2d(1024, 768, 1, 1, 0),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.Conv2d(256, 128, 1, 1, 0),
        ])
        self.sub_pixel_convs = nn.ModuleList([
            nn.PixelShuffle(2),
            nn.PixelShuffle(2),
            nn.PixelShuffle(2)
        ])
        self.spatial_attention = nn.ModuleList([
            SpatialAttentionModule(),
            SpatialAttentionModule(),
            SpatialAttentionModule()
        ])
        self.attentions = nn.ModuleList([
            SelfAttention(1024),
            SelfAttention(512),
            SelfAttention(256)
        ])
        self.co_attentions = nn.ModuleList([
            CoordAtt(768, 768, reduction=32),
            CoordAtt(256, 256, reduction=32),
            CoordAtt(128, 128, reduction=32),
            # CrissCrossAttention(768),
            # CrissCrossAttention(256),
            # CrissCrossAttention(128)
        ])

    def forward(self, features):
        p3, p4, p5, p6, p7, p8 = features
        p5 = self.attentions[0](torch.cat([self.spatial_attention[0](self.sub_pixel_convs[0](self.up_channels(p6))),
                                           self.co_attentions[0](self.down_channels[0](p5))], dim=1))
        p4 = self.attentions[1](torch.cat([self.spatial_attention[1](self.sub_pixel_convs[1](p5)[:, :, :-1, :-1]),
                                           self.co_attentions[1](self.down_channels[1](p4))], dim=1))
        p3 = self.attentions[2](torch.cat([self.spatial_attention[2](self.sub_pixel_convs[2](p4)),
                                          self.co_attentions[2](self.down_channels[2](p3))], dim=1))
        return p3, p4, p5, p6, p7, p8


class SSDNet(nn.Module):
    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSDNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size

        if size == 300:
            self.indicator = 3
        elif size == 512:
            self.indicator = 5
        else:
            print("Error: Sorry only SSD300 and SSD512 are supported!")
            return
        # vgg network
        self.base = base
        # conv_4
        self.extras = SA_down()
        self.dsf = DSF()
        self.uf = Up_Fusion()
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        sources = self.base(x)
        sources = list(self.extras(tuple(sources)))
        sources[3] = self.dsf(tuple(sources[-3:]))
        sources = list(self.uf(tuple(sources)))
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())


class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels / 2)

        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out


class Darknet53(nn.Module):
    def __init__(self, block=DarkResidualBlock):
        super(Darknet53, self).__init__()

        self.conv1 = conv_batch(3, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.residual_block1 = self.make_layer(block, in_channels=64, num_blocks=1)
        self.conv3 = conv_batch(64, 128, stride=2)
        self.residual_block2 = self.make_layer(block, in_channels=128, num_blocks=2)
        self.conv4 = conv_batch(128, 256, stride=2)
        self.residual_block3 = self.make_layer(block, in_channels=256, num_blocks=8)
        self.conv5 = conv_batch(256, 512, stride=2)
        self.residual_block4 = self.make_layer(block, in_channels=512, num_blocks=8)
        self.conv6 = conv_batch(512, 1024, stride=2)
        self.residual_block5 = self.make_layer(block, in_channels=1024, num_blocks=4)

    def forward(self, x):
        sources = list()
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        # sources.append(out)
        out = self.conv4(out)
        out = self.residual_block3(out)
        sources.append(out)
        out = self.conv5(out)
        out = self.residual_block4(out)
        sources.append(out)
        out = self.conv6(out)
        out = self.residual_block5(out)
        sources.append(out)
        return sources

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}


def multibox(darknet, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    sources_channels = [256, 512, 1024, 256, 256, 256]
    for n_box, channel in zip(cfg, sources_channels):
        loc_layers += [nn.Conv2d(channel, n_box * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(channel, n_box * num_classes, kernel_size=3, padding=1)]
    return darknet, extra_layers, (loc_layers, conf_layers)


mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}


def build_net(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300 and size != 512:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    return SSDNet(phase,
                  size,
                  *multibox(Darknet53(),
                            None,
                            mbox[str(size)], num_classes),
                  num_classes)
