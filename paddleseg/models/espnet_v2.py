import copy
import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager, param_init
from paddleseg import utils

class ConvBNPRelu(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding='same',
        dilation=1,
        groups=1
        ):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias_attr=False
            )
        self.bn = nn.BatchNorm2D(out_channels)
        self.act = nn.PReLU(out_channels)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        # output = self.conv1(output)
        output = self.bn(output)
        output = self.act(output)
        return output

class ConvBN(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding='same',
        dilation=1,
        groups=1
        ):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias_attr=False
            )
        self.bn = nn.BatchNorm2D(out_channels)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        # output = self.conv1(output)
        output = self.bn(output)
        return output


class Conv(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding='same',
        dilation=1,
        groups=1
        ):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias_attr=False
            )

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output


class PSPModule(nn.Layer):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 4, 8)):
        super().__init__()
        self.stages = nn.LayerList([
            copy.deepcopy(Conv(features, features, 3, 1, groups=features))
            for size in sizes])
        self.project = ConvBNPRelu(features * (len(sizes) + 1), out_features, 1, 1)
 
    def forward(self, feats):
        h, w = feats.shape[2], feats.shape[3]
        out = [feats]
        for stage in self.stages:
            feats = F.avg_pool2d(feats, kernel_size=3, stride=2, padding=1)
            upsampled = F.interpolate(stage(feats), size=(h, w), mode='bilinear', align_corners=True)
            out.append(upsampled)
        return self.project(paddle.concat(out, axis=1))


class EESP(nn.Layer):
    def __init__(self, in_channels, out_channels, stride=1, k=4, r_lim=7, down_method='esp'):
        super().__init__()
        self.out_channels = out_channels
        self.stride = stride
        n = int(out_channels / k)
        n1 = out_channels - (k - 1) * n
        assert down_method in ['avg', 'esp'], 'One of these is suppported (avg or esp)'
        assert n == n1, "n(={}) and n1(={}) should be equal for Depth-wise Convolution ".format(n, n1)
        self.proj_1x1 = ConvBNPRelu(in_channels, n, 1, groups=k)
        
        map_receptive_ksize = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8}
        self.k_sizes = []
        for i in range(k):
            ksize = int(3 + 2 * i)
            # After reaching the receptive field limit, fall back to the base kernel size of 3 with a dilation rate of 1
            ksize = ksize if ksize <= r_lim else 3
            self.k_sizes.append(ksize)
        self.k_sizes.sort()

        self.spp_dw = nn.LayerList()

        for i in range(k):
            d_rate = map_receptive_ksize[self.k_sizes[i]]
            self.spp_dw.append(Conv(n, n, 3, stride=stride, groups=n, dilation=d_rate))

        self.conv_1x1_exp = ConvBN(out_channels, out_channels, 1, 1,groups=k)
        self.br_after_cat = nn.Sequential(
            nn.BatchNorm2D(out_channels),
            nn.PReLU(out_channels)
        )
        self.module_act = nn.PReLU(out_channels)
        self.downAvg = True if down_method == 'avg' else False

    def forward(self, x):

        output1 = self.proj_1x1(x)
        outputs = [self.spp_dw[0](output1)]
        for k in range(1,len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            out_k = out_k + outputs[-1]
            outputs.append(out_k)
        # Merge
        expanded = self.conv_1x1_exp( # Aggregate the feature maps using point-wise convolution
            self.br_after_cat( # apply batch normalization followed by activation function (PRelu in this case)
                paddle.concat(outputs, 1) # concatenate the output of different branches
            )
        )
        del outputs
        # if down-sampling, then return the concatenated vector
        # as Downsampling function will combine it with avg. pooled feature map and then threshold it
        if self.stride == 2 and self.downAvg:
            return expanded

        # if dimensions of input and concatenated vector are the same, add them (RESIDUAL LINK)
        if expanded.shape == x.shape:
            expanded = expanded + x

        # Threshold the feature map using activation function (PReLU in this case)
        return self.module_act(expanded)


class DownSampler(nn.Layer):
    '''
    Down-sampling fucntion that has two parallel branches: (1) avg pooling
    and (2) EESP block with stride of 2. The output feature maps of these branches
    are then concatenated and thresholded using an activation function (PReLU in our
    case) to produce the final output.
    '''

    def __init__(self, in_channels, out_channels, k=4, r_lim=9, reinf=True):
        '''
            :param nin: number of input channels
            :param nout: number of output channels
            :param k: # of parallel branches
            :param r_lim: A maximum value of receptive field allowed for EESP block
            :param g: number of groups to be used in the feature map reduction step.
        '''
        super().__init__()
        self.out_channels = out_channels
        out_new = out_channels - in_channels
        self.eesp = EESP(in_channels, out_new, stride=2, k=k, r_lim=r_lim, down_method='avg')
        self.avg = nn.AvgPool2D(kernel_size=3, padding=1, stride=2)
        if reinf:
            self.inp_reinf = nn.Sequential(
                ConvBNPRelu(config_inp_reinf, config_inp_reinf, 3, 1),
                ConvBN(config_inp_reinf, out_channels, 1, 1)
            )
        self.act =  nn.PReLU(out_channels)

    def forward(self, input, input2=None):
        '''
        :param input: input feature map
        :return: feature map down-sampled by a factor of 2
        '''
        avg_out = self.avg(input)
        eesp_out = self.eesp(input)
        output = paddle.concat([avg_out, eesp_out], 1)
        if input2 is not None:
            #assuming the input is a square image
            w1 = avg_out.shape[2]
            while True:
                input2 = F.avg_pool2d(input2, kernel_size=3, padding=1, stride=2)
                w2 = input2.shape[2]
                if w2 == w1:
                    break
            output = output + self.inp_reinf(input2)

        return self.act(output) #self.act(output)


class EESPNet(nn.Layer):
    '''
    the ESPNetv2 implementation for backbone
    '''

    def __init__(self, s=1):
        '''
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param s: factor that scales the number of output feature maps
        '''
        super().__init__()
        reps = [0, 3, 7, 3]  # how many times EESP blocks should be repeated.
        channels = 3

        r_lim = [13, 11, 9, 7, 5]  # receptive field at each spatial level
        K = [4]*len(r_lim) # No. of parallel branches at different levels

        base = 32 #base configuration
        config_len = 5
        config = [base] * config_len
        base_s = 0
        for i in range(config_len):
            if i== 0:
                base_s = int(base * s)
                base_s = math.ceil(base_s / K[0]) * K[0]
                config[i] = base if base_s > base else base_s
            else:
                config[i] = base_s * pow(2, i)
        if s <= 1.5:
            config.append(1024)
        elif s in [1.5, 2]:
            config.append(1280)
        else:
            ValueError('Configuration not supported')

        #print('Config: ', config)

        global config_inp_reinf
        config_inp_reinf = 3
        self.input_reinforcement = True
        assert len(K) == len(r_lim), 'Length of branching factor array and receptive field array should be the same.'
        
        self.level1 = ConvBNPRelu(channels, config[0], 3, 2)  # 112 L1

        self.level2_0 = DownSampler(config[0], config[1], k=K[0], r_lim=r_lim[0], reinf=self.input_reinforcement)  # out = 56
        self.level3_0 = DownSampler(config[1], config[2], k=K[1], r_lim=r_lim[1], reinf=self.input_reinforcement) # out = 28
        self.level3 = nn.LayerList([
            copy.deepcopy(EESP(config[2], config[2], stride=1, k=K[2], r_lim=r_lim[2]))
            for _ in range(reps[1])
        ])

        self.level4_0 = DownSampler(config[2], config[3], k=K[2], r_lim=r_lim[2], reinf=self.input_reinforcement) #out = 14
        self.level4 = nn.LayerList()
        self.level4 = nn.LayerList([
            copy.deepcopy(EESP(config[3], config[3], stride=1, k=K[3], r_lim=r_lim[3]))
            for _ in range(reps[2])
        ])

    def forward(self, input, p=0.2, seg=True):
        '''
        :param input: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''
        out_l1 = self.level1(input)  # 112
        if not self.input_reinforcement:
            del input
            input = None

        out_l2 = self.level2_0(out_l1, input)  # 56

        out_l3_0 = self.level3_0(out_l2, input)  # out_l2_inp_rein
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)

        out_l4_0 = self.level4_0(out_l3, input)  # down-sampled
        for i, layer in enumerate(self.level4):
            if i == 0:
                out_l4 = layer(out_l4_0)
            else:
                out_l4 = layer(out_l4)
        return out_l1, out_l2, out_l3, out_l4


@manager.MODELS.add_component
class EESPNet_Seg(nn.Layer):
    """
    espnetv2 implementation for segmentation
    """
    def __init__(self, num_classes=20, s=1, pretrained=None, pretrained_backbone=None):
        super().__init__()
        self.pretrained = pretrained
        self.net = EESPNet(s=s)
        if pretrained_backbone is not None:
            self.net.set_state_dict(paddle.load(pretrained_backbone))
        if s <=0.5:
            p = 0.1
        else:
            p=0.2

        self.proj_L4_C = ConvBNPRelu(self.net.level4[-1].out_channels, self.net.level3[-1].out_channels, 1, 1)
        pspSize = 2*self.net.level3[-1].out_channels
        self.pspMod = nn.Sequential(EESP(pspSize, pspSize //2, stride=1, k=4, r_lim=7),
                PSPModule(pspSize // 2, pspSize //2))
        self.project_l3 = nn.Sequential(
            nn.Dropout2D(p=p),
            Conv(pspSize // 2, num_classes, 1, 1)
        )
        self.act_l3 = nn.Sequential(
            nn.BatchNorm2D(num_classes),
            nn.PReLU(num_classes)
        )
        self.project_l2 = ConvBNPRelu(self.net.level2_0.out_channels + num_classes, num_classes, 1, 1)
        self.project_l1 = nn.Sequential(
            nn.Dropout2D(p=p),
            Conv(self.net.level1.out_channels + num_classes, num_classes, 1, 1)
        )
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def hierarchicalUpsample(self, x, factor=3):
        for i in range(factor):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x


    def forward(self, input):
        out_l1, out_l2, out_l3, out_l4 = self.net(input, seg=True)
        out_l4_proj = self.proj_L4_C(out_l4)
        up_l4_to_l3 = F.interpolate(out_l4_proj, scale_factor=2, mode='bilinear', align_corners=True)
        merged_l3_upl4 = self.pspMod(paddle.concat([out_l3, up_l4_to_l3], 1))
        proj_merge_l3_bef_act = self.project_l3(merged_l3_upl4)
        proj_merge_l3 = self.act_l3(proj_merge_l3_bef_act)
        out_up_l3 = F.interpolate(proj_merge_l3, scale_factor=2, mode='bilinear', align_corners=True)
        merge_l2 = self.project_l2(paddle.concat([out_l2, out_up_l3], 1))
        out_up_l2 = F.interpolate(merge_l2, scale_factor=2, mode='bilinear', align_corners=True)
        merge_l1 = self.project_l1(paddle.concat([out_l1, out_up_l2], 1))
        if self.training:
            return F.interpolate(merge_l1, scale_factor=2, mode='bilinear', align_corners=True), self.hierarchicalUpsample(proj_merge_l3_bef_act)
        else:
            return [F.interpolate(merge_l1, scale_factor=2, mode='bilinear', align_corners=True)]
