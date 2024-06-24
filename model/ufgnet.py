import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BasicConv2d import BasicConv2d
from models.RMFE import RC3
from math import log
from net.pvtv2 import pvt_v2_b2
# from models.HolisticAttention import HA
from models.SEA import MSCA
from torch.nn.parameter import Parameter
from models.LaPlacianMs import LaPlacianMs
from thop import profile
import numpy as np

class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Spade(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(Spade, self).__init__()
        self.param_free_norm = nn.BatchNorm2d(out_channels, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.mlp_gamma = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, edge):
        normalized = self.param_free_norm(x)

        edge = F.interpolate(edge, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(edge)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out

    def initialize(self):
        weight_init(self)

def upsample_list(pred_list, target_size):
    for i in range(len(pred_list)):
        pred_list[i] = upsample(pred_list[i], target_size)
    return pred_list


def upsample(tensor, size):
    return F.interpolate(tensor, size, mode='bilinear', align_corners=True)


class ConvBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)





class BFM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BFM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0))
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, (3, 1), padding=(1, 0))
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, (3, 1), padding=(1, 0))
        )

        self.branch4 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, (3, 1), padding=(1, 0))
        )

        self.conv = nn.Conv2d(in_channel, out_channel, 1)

        self.conv_cat = nn.Conv2d(out_channel * 4, out_channel, 3, padding=1)


        # self.foreground_conv = nn.Conv2d(sc_channel, sc_channel, 3, 1, 1, bias=False)
        self.conv1 = BasicConv2d(64, out_channel, 3, 1, 1, relu=True)
        self.comb_conv0 = BasicConv2d(out_channel * 2, out_channel, 3, 1, 1, relu=True)
        self.comb_conv1 = BasicConv2d(out_channel * 2, out_channel, 3, 1, 1, relu=True)
        self.comb_conv2= BasicConv2d(out_channel * 2, out_channel, 3, 1, 1, relu=True)
        self.comb_conv3 = BasicConv2d(out_channel * 2, out_channel, 3, 1, 1, relu=True)
        self.comb_conv4 = BasicConv2d(out_channel * 2, out_channel, 3, 1, 1, relu=True)
        self.conv2d = BasicConv2d(out_channel, out_channel, 3, 1, 1, relu=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

        self.loc_map_conv = nn.Conv2d(out_channel, 1, 3, 1, 1)

    def forward(self, low, rf_conv):

        crf_conv = self.conv1(rf_conv)
        loc_map = self.loc_map_conv(crf_conv)
        loc_map = upsample(loc_map, low.shape[2:])
        loc_map = torch.sigmoid(loc_map)

        x0 = self.branch0(low)
        foreground = x0 * loc_map
        background = x0 * (1 - loc_map)

        comb_conv0 = self.comb_conv0(torch.cat((foreground, background), dim=1))
        x1 = self.branch1(low)
        foreground1 = x1 * loc_map
        background1 = x1 * (1 - loc_map)
        comb_conv1 = self.comb_conv1(torch.cat((foreground1, background1), dim=1))
        x2 = self.branch2(self.conv(low) + comb_conv1)
        foreground2 = x2 * loc_map
        background2 = x2 * (1 - loc_map)
        comb_conv2 = self.comb_conv2(torch.cat((foreground2, background2), dim=1))
        x3 = self.branch3(self.conv(low) + comb_conv2)
        foreground3 = x3 * loc_map
        background3 = x3 * (1 - loc_map)
        comb_conv3 = self.comb_conv3(torch.cat((foreground3, background3), dim=1))
        x4 = self.branch4(self.conv(low) + comb_conv3)
        foreground4 = x4 * loc_map
        background4 = x4 * (1 - loc_map)
        comb_conv4 = self.comb_conv4(torch.cat((foreground4, background4), dim=1))

        x_cat = self.conv_cat(torch.cat((comb_conv1, comb_conv2, comb_conv3, comb_conv4), dim=1))
        x = self.relu(comb_conv0 + x_cat)

        return x


class RecFusion_S1(nn.Module):
    def __init__(self, channel):
        super(RecFusion_S1, self).__init__()

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_x2cbr = BasicConv2d(channel * 2, channel, 3, padding=1)
        self.conv_x3cbr = BasicConv2d(channel * 2, channel, 3, padding=1)
        self.conv_x0cbr = BasicConv2d(channel * 2, channel, 3, padding=1)
        self.iter_x1cbr = BasicConv2d(channel * 2, channel, 3, padding=1)
        self.out_cbr = BasicConv2d(channel * 2, channel, 3, padding=1)
        self.at_conv1 = BasicConv2d(channel, channel, 3, 1, 2, dilation=2, relu=True)
        self.at_conv2 = BasicConv2d(channel, channel, 3, 1, 3, dilation=3, relu=True)
        self.at_it1 = BasicConv2d(channel, channel, 3, 1, 2, dilation=2, relu=True)
        self.at_it2 = BasicConv2d(channel, channel, 3, 1, 3, dilation=3, relu=True)
        self.at_conv3 = BasicConv2d(channel, channel, 3, 1, 2, dilation=2, relu=True)
        self.at_conv4 = BasicConv2d(channel, channel, 3, 1, 3, dilation=3, relu=True)
        self.at_conv5 = BasicConv2d(channel, channel, 3, 1, 2, dilation=2, relu=True)
        self.at_conv6 = BasicConv2d(channel, channel, 3, 1, 3, dilation=3, relu=True)
        self.theta = nn.Conv2d(64, 64, 1)
        self.theta2 = nn.Conv2d(64, 64, 1)
        self.theta3 = nn.Conv2d(64, 64, 1)
        self.theta4 = nn.Conv2d(64, 64, 1)
        # self.weight = nn.Parameter(torch.randn(1, 128, 1, 1))
        nn.init.constant_(self.theta.weight, 1)
        nn.init.constant_(self.theta2.weight, 1)
        nn.init.constant_(self.theta3.weight, 1)
        nn.init.constant_(self.theta4.weight, 1)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self,  x3, x2, x1, iternum=3):
        out = []
        # size = zt4_e.size()[2:]
        up_x3 = self.upsample(x3)
        up_x2 = self.upsample1(x2)

        x2_cbr = torch.cat((up_x3, up_x2), dim=1)
        x2_cbr = self.conv_x2cbr(x2_cbr)
        fd1 = self.at_conv1(x2_cbr)
        fd2 = self.at_conv2(x2_cbr)
        cm_fd = x2_cbr + fd1 + fd2

        x1_cbr = torch.cat((x1, cm_fd), dim=1)
        iter0 = self.iter_x1cbr(x1_cbr)
        at_it1 = self.at_it1(iter0)
        at_it2 = self.at_it2(iter0)
        cm_it = iter0 + at_it1 + at_it2

        out.append(cm_it)

        for _ in range(1, iternum):

            x3_cbr = torch.cat((up_x3, cm_it), dim=1)
            x3_cbr = self.conv_x3cbr(x3_cbr)
            fd3 = self.at_conv3(x3_cbr)
            fd4 = self.at_conv4(x3_cbr)
            cm_fd3 = x3_cbr + fd3 + fd4

            x2_cbr = torch.cat((up_x2, cm_fd3), dim=1)
            x2_cbr = self.conv_x2cbr(x2_cbr)
            fd1 = self.at_conv1(x2_cbr)
            fd2 = self.at_conv2(x2_cbr)
            cm_fd = x2_cbr + fd1 + fd2

            x0_cbr = torch.cat((x1, cm_it), dim=1)
            x0_cbr = self.conv_x0cbr(x0_cbr)
            fd5 = self.at_conv5(x0_cbr)
            fd6 = self.at_conv6(x0_cbr)
            cm_fd0 = x0_cbr + fd5 + fd6

            iter_cbr = torch.cat((cm_fd0, cm_fd), dim=1)
            iter0 = self.iter_x1cbr(iter_cbr)
            at_it1 = self.at_it1(iter0)
            at_it2 = self.at_it2(iter0)
            cm_it = iter0 + at_it1 + at_it2

            out.append(cm_it)

        return out
class MBR(nn.Module):
    expansion =1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(MBR, self).__init__()
        # branch1
        t = int(abs((log(64, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample1 = upsample
        self.stride1 = stride
        # barch2
        self.conv3 = nn.Conv2d(inplanes, inplanes, kernel_size=5, stride=1,
                               padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.relu3 = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv4 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv4 = nn.Conv2d(inplanes, planes, kernel_size=5, stride=stride,
                                   padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.conv_cat = nn.Conv2d(2 * inplanes, inplanes, 3, 1, 1)
        self.upsample2 = upsample
        self.stride2 = stride
        self.conv2d = BasicConv2d(16, 16, 3, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # if x.size() != att.size():
        #     att = F.interpolate(att, x.size()[2:], mode='bilinear', align_corners=False)
        residual = x

        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        # out1 = out1 * att + out1
        out1 = self.conv2(out1)
        out1 = self.bn2(out1)

        if self.upsample1 is not None:
            residual = self.upsample1(x)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out2 = self.relu3(out2)

        # out2 = out2 * att + out2
        out2 = self.conv4(out2)
        out2 = self.bn4(out2)

        if self.upsample2 is not None:
            residual = self.upsample2(x)
        out = self.conv_cat(torch.cat((out1, out2), 1))
        out += residual
        out = self.relu(out)
        x_out = self.conv2d(out)
        wei = self.avg_pool(x_out)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x_out = x_out * wei

        return x_out

class RecFusion_S2(nn.Module):
    def __init__(self, channel):
        super(RecFusion_S2, self).__init__()

        self.conv_x2cbr = BasicConv2d(channel * 2, channel, 3, padding=1)
        self.conv_x0cbr = BasicConv2d(channel * 2, channel, 3, padding=1)
        self.iter_x1cbr = BasicConv2d(channel * 2, channel, 3, padding=1)
        self.at_it1 = BasicConv2d(channel, channel, 3, 1, 2, dilation=2, relu=True)
        self.at_it2 = BasicConv2d(channel, channel, 3, 1, 3, dilation=3, relu=True)
        self.at_conv3 = BasicConv2d(channel, channel, 3, 1, 2, dilation=2, relu=True)
        self.at_conv4 = BasicConv2d(channel, channel, 3, 1, 3, dilation=3, relu=True)
        self.at_conv5 = BasicConv2d(channel, channel, 3, 1, 2, dilation=2, relu=True)
        self.at_conv6 = BasicConv2d(channel, channel, 3, 1, 3, dilation=3, relu=True)
        self.theta2 = nn.Conv2d(64, 64, 1)
        self.theta3 = nn.Conv2d(64, 64, 1)
        self.theta4 = nn.Conv2d(64, 64, 1)
        nn.init.constant_(self.theta2.weight, 1)
        nn.init.constant_(self.theta3.weight, 1)
        nn.init.constant_(self.theta4.weight, 1)
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x2, x1, iternum=2):
        out = []
        up_x2 = self.upsample(x2)

        x1_cbr = torch.cat((x1, up_x2), dim=1)
        iter0 = self.iter_x1cbr(x1_cbr)
        at_it1 = self.at_it1(iter0)
        at_it2 = self.at_it2(iter0)
        cm_it = iter0 + at_it1 + at_it2

        out.append(cm_it)

        for _ in range(1, iternum):

            x2_cbr = torch.cat((up_x2, cm_it), dim=1)
            x2_cbr = self.conv_x2cbr(x2_cbr)
            fd3 = self.at_conv3(x2_cbr)
            fd4 = self.at_conv4(x2_cbr)
            cm_fd2 = x2_cbr + fd3 + fd4

            x0_cbr = torch.cat((x1, cm_fd2), dim=1)
            x0_cbr = self.conv_x0cbr(x0_cbr)
            fd5 = self.at_conv5(x0_cbr)
            fd6 = self.at_conv6(x0_cbr)
            cm_fd0 = x0_cbr + fd5 + fd6

            iter_cbr = torch.cat((cm_fd0, cm_fd2), dim=1)
            iter0 = self.iter_x1cbr(iter_cbr)
            at_it1 = self.at_it1(iter0)
            at_it2 = self.at_it2(iter0)
            cm_it = iter0 + at_it1 + at_it2

            out.append(cm_it)

        return out
    def feature_grouping(self, xr, xg, M):
        if M == 1:
            q = torch.cat(
                (xr, xg), 1)
        elif M == 2:
            xr_g = torch.chunk(xr, 2, dim=1)
            xg_g = torch.chunk(xg, 2, dim=1)
            q = torch.cat(
                (xr_g[0], xg_g[0], xr_g[1], xg_g[1]), 1)
        elif M == 4:
            xr_g = torch.chunk(xr, 4, dim=1)
            xg_g = torch.chunk(xg, 4, dim=1)
            q = torch.cat(
                (xr_g[0], xg_g[0], xr_g[1], xg_g[1], xr_g[2], xg_g[2], xr_g[3], xg_g[3]), 1)
        elif M == 8:
            xr_g = torch.chunk(xr, 8, dim=1)
            xg_g = torch.chunk(xg, 8, dim=1)
            q = torch.cat(
                (xr_g[0], xg_g[0], xr_g[1], xg_g[1], xr_g[2], xg_g[2], xr_g[3], xg_g[3],
                 xr_g[4], xg_g[4], xr_g[5], xg_g[5], xr_g[6], xg_g[6], xr_g[7], xg_g[7]), 1)
        elif M == 16:
            xr_g = torch.chunk(xr, 16, dim=1)
            xg_g = torch.chunk(xg, 16, dim=1)
            q = torch.cat(
                (xr_g[0], xg_g[0], xr_g[1], xg_g[1], xr_g[2], xg_g[2], xr_g[3], xg_g[3],
                 xr_g[4], xg_g[4], xr_g[5], xg_g[5], xr_g[6], xg_g[6], xr_g[7], xg_g[7],
                 xr_g[8], xg_g[8], xr_g[9], xg_g[9], xr_g[10], xg_g[10], xr_g[11], xg_g[11],
                 xr_g[12], xg_g[12], xr_g[13], xg_g[13], xr_g[14], xg_g[14], xr_g[15], xg_g[15]), 1)
        elif M == 32:
            xr_g = torch.chunk(xr, 32, dim=1)
            xg_g = torch.chunk(xg, 32, dim=1)
            q = torch.cat(
                (xr_g[0], xg_g[0], xr_g[1], xg_g[1], xr_g[2], xg_g[2], xr_g[3], xg_g[3],
                 xr_g[4], xg_g[4], xr_g[5], xg_g[5], xr_g[6], xg_g[6], xr_g[7], xg_g[7],
                 xr_g[8], xg_g[8], xr_g[9], xg_g[9], xr_g[10], xg_g[10], xr_g[11], xg_g[11],
                 xr_g[12], xg_g[12], xr_g[13], xg_g[13], xr_g[14], xg_g[14], xr_g[15], xg_g[15],
                 xr_g[16], xg_g[16], xr_g[17], xg_g[17], xr_g[18], xg_g[18], xr_g[19], xg_g[19],
                 xr_g[20], xg_g[20], xr_g[21], xg_g[21], xr_g[22], xg_g[22], xr_g[23], xg_g[23],
                 xr_g[24], xg_g[24], xr_g[25], xg_g[25], xr_g[26], xg_g[26], xr_g[27], xg_g[27],
                 xr_g[28], xg_g[28], xr_g[29], xg_g[29], xr_g[30], xg_g[30], xr_g[31], xg_g[31]), 1)
        else:
            raise Exception("Invalid Group Number!")

        return q

BN_MOMENTUM = 0.01
class MVSSNet(nn.Module):
    def __init__(self, nclass=64, n_input=3, **kwargs):
        super(MVSSNet, self).__init__()
        # self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.inplanes = 16
        # uncertainty related
        kernel = torch.ones((7, 7))
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.hidden_dim = 512
        self.input_proj = nn.Sequential(ConvBR(512, self.hidden_dim, kernel_size=1), nn.Dropout2d(p=0.1))
        self.conv = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1, bias=False)

        self.mean_conv = nn.Conv2d(self.hidden_dim, 1, kernel_size=1, bias=False)
        self.std_conv = nn.Conv2d(self.hidden_dim, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=False)
        self.RME4 = RC3(512, 64)
        # frequency branch
        self.conv1fre = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1fre = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2fre = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2fre = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv3fre = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3fre = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv4fre = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4fre = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv5fre = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5fre = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.laplacian = LaPlacianMs(in_c=64, gauss_ker_size=3, scale=[2, 4, 8])

        self.spade = Spade(64, 64)
        self.fre_conv = nn.Conv2d(64, 1, 3, 1, 1)
        self.conv_cat = nn.Conv2d(512 * 2, 512, 3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.conv_concat = BasicConv2d(3 * 64, 64, 3, padding=1, relu=True)
        self.conv_concat2 = BasicConv2d(2 * 64, 64, 3, padding=1, relu=True)
        self.reduce_rec = BasicConv2d(64, self.inplanes, 3, 1, 1)
        self.bsp0 = BFM(512, 64)
        self.bsp3 = BFM(320, 64)
        self.bsp2 = BFM(128, 64)
        self.bsp1 = BFM(64, 64)
        self.bsp4 = BFM(64, 64)
        self.upsample0 = nn.Sequential(
            nn.ConvTranspose2d(self.inplanes, self.inplanes,
                               kernel_size=2, stride=2,
                               padding=0, bias=False),
            nn.BatchNorm2d(self.inplanes),
        )

        # self.rec = RecFusion_S1(64)
        # self.rec2 = RecFusion_S2(64)
        self.reduce_rec = BasicConv2d(64, self.inplanes, 3, 1, 1)
        self.deconv1 = MBR(self.inplanes, 16, 2, self.upsample0)
        self.gui_linearr = nn.Conv2d(16, 1, 3, 1, padding=1)

        self.initialize()

    def reparameterize(self, mu, logvar, k=1):
        sample_z = []
        for _ in range(k):
            std = logvar.mul(0.5).exp_()  # type: Variable # 1, 1, 33, 33
            # eps = std.data.new(std.size()).normal_() # 1, 1, 33, 33  fill with gaussion N(0, 1); change every loop
            eps = np.float32(np.random.laplace(0, 1, std.size()))
            eps = torch.from_numpy(eps).cuda()
            sample_z.append(eps.mul(std).add_(mu))
        sample_z = torch.cat(sample_z, dim=1)
        return sample_z
    # def reparameterize(self, mu, logvar, k=1, eps=1e-10):
    #     sample_z = []
    #     for _ in range(k):
    #         # 计算尺度参数beta，这里假设logvar代表log(sigma^2)，所以beta = sqrt(sigma^2)
    #         beta = logvar.mul(0.5).exp_()  # 注意：确保logvar的值不会导致溢出
    #
    #         # 生成(0,1)上的均匀分布样本，并加上一个小的正数eps防止log(0)
    #         U = beta.data.new(beta.size()).uniform_()  # 生成U
    #         U = U.clamp(min=eps, max=1 - eps)  # 避免log(0)导致的NaN
    #
    #         # 应用Gumbel分布的公式
    #         gumbel_sample = -torch.log(-torch.log(U))
    #
    #         # 生成的样本乘以尺度参数beta并加上位置参数mu
    #         sample_z.append(gumbel_sample.mul(beta).add_(mu))
    #     sample_z = torch.cat(sample_z, dim=1)
    #     return sample_z
    def forward(self, x):
        size = x.size()[2:]
        input_ = x.clone()

        # c1, c2, c3, c4 = self.resnet(input_)
        pvt = self.backbone(input_)

        x_fre = self.conv1fre(x)
        x_fre = self.bn1fre(x_fre)
        x_fre = self.relu(x_fre)
        x_fre = self.laplacian(x_fre)
        # fre_map = self.fre_conv(x_fre)
        x_fre = self.conv2fre(x_fre)
        x_fre = self.bn2fre(x_fre)
        x_fre = self.relu(x_fre)
        x_fre = self.conv3fre(x_fre)
        x_fre = self.bn3fre(x_fre)
        x_fre = self.relu(x_fre)
        x_fre = self.conv4fre(x_fre)
        x_fre = self.bn4fre(x_fre)
        x_fre = self.relu(x_fre)
        x_fre = self.conv5fre(x_fre)
        x_fre = self.bn5fre(x_fre)
        x_fre = self.relu(x_fre)
        # x_fre = self.FE(x_fre)
        c1 = pvt[0]
        c2 = pvt[1]
        c3 = pvt[2]
        c4 = pvt[3]

        x_u = self.input_proj(c4)  # 1, 512, 33, 33
        # residual = self.conv(x_u)  # 1, 512, 33, 33
        mean = self.mean_conv(x_u)  # 1, 1, 33, 33
        std = self.std_conv(x_u)  # 1, 1, 33, 33

        prob_x = self.reparameterize(mean, std, 1)
        prob_out2 = self.reparameterize(mean, std, 50)  # 1, 50, 33, 33 sample process
        prob_out2 = torch.sigmoid(prob_out2)  # 1, 50, 33, 33

        uncertainty = prob_out2.var(dim=1, keepdim=True).detach()  # 1, 1, 33, 33
        if self.training:
            uncertainty = F.conv2d(uncertainty, self.weight, padding=3, groups=1)
            uncertainty = F.conv2d(uncertainty, self.weight, padding=3, groups=1)
            uncertainty = F.conv2d(uncertainty, self.weight, padding=3, groups=1)
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())

        u_rme4 = self.spade(x_fre, uncertainty)

        rf0 = self.bsp0(c4, u_rme4)
        rf3 = self.bsp3(c3, rf0)
        rf2 = self.bsp2(c2, rf3)
        rf1 = self.bsp1(c1, rf2)

        up_x3 = self.upsample(rf3)
        up_x2 = self.upsample1(rf2)
        ag0 = torch.cat((rf1, up_x2, up_x3), dim=1)
        ag = self.conv_concat(ag0)
        rf4 = self.bsp4(rf0, ag)

        # cbrf = self.rec2(rf4, ag)
        rec_f = self.conv_concat2(torch.cat((self.upsample2(rf4), ag), dim=1))
        # e_rf = self.deconv1(rec_f)
        rec_f = self.reduce_rec(rec_f)
        e_rf = self.deconv1(rec_f)
        gui_mp = self.gui_linearr(e_rf)
        # gui_mp = self.gui_linearr(rec_f)
        Sig_gui_mp = torch.sigmoid(gui_mp)

        return upsample(Sig_gui_mp, size), upsample(prob_x, size), upsample(uncertainty, size)

    def initialize(self):
        weight_init(self)

    def _make_CIM(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            print("--------------")
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            print("*******************")
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
            if m.weight is None:
                pass
            elif m.bias is not None:
                nn.init.zeros_(m.bias)
            else:
                nn.init.ones_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.ReLU6, nn.Upsample, Parameter, nn.AdaptiveAvgPool2d, nn.Sigmoid)):
            pass


def get_mvss(backbone='res2net50', pretrained_base=True, nclass=64, sobel=True, n_input=3,  **kwargs):
    model = MVSSNet(nclass, backbone=backbone,
                    pretrained_base=pretrained_base,
                    sobel=sobel,
                    n_input=n_input,
                    **kwargs)
    return model


if __name__ == '__main__':
    img = torch.randn(1, 3, 416, 416)
    img = img.cuda()
    model = get_mvss(n_input=3, constrain=True)
    model = model.cuda()
    with torch.no_grad():
        out = model(img)
    flops, params = profile(model, inputs=(img,), verbose=False)

    print(f"FLOPs: {flops}, Params: {params}")
    # edge, outputs= model(img)
    # # print(mp0.shape)
    # print(outputs.shape)
    # print(edge.shape)

    # edge, outputs= model(img)
    # # print(mp0.shape)
    # print(outputs.shape)
    # print(edge.shape)


