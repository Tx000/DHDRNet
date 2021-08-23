import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from dcn.deform_conv import ModulatedDeformConvPack as DCN
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')


class DCN_off_b(nn.Module):
    def __init__(self, filters, groups):
        super(DCN_off_b, self).__init__()
        self.conv = nn.Conv2d(filters * 2, filters * 2, 3, 1, 1, bias=True)
        self.conv_offset = nn.Conv2d(filters * 2, filters, 3, 1, 1, bias=True)
        self.conv_x = nn.Conv2d(filters * 2, filters, 3, 1, 1, bias=True)
        self.conv_dcnpack = DCN(filters, filters, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                                extra_offset_mask=True)
        self.lrule = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, ali, ref):
        tensor = self.lrule(self.conv(torch.cat([ali, ref], dim=1)))
        offset = self.lrule(self.conv_offset(tensor))
        x = self.lrule(self.conv_x(tensor))
        output = self.lrule(self.conv_dcnpack([x, offset]))
        return output, offset


class DCN_off(nn.Module):
    def __init__(self, filters, groups):
        super(DCN_off, self).__init__()
        self.conv = nn.Conv2d(filters * 2, filters * 2, 3, 1, 1, bias=True)
        self.conv_offset_1 = nn.Conv2d(filters * 2, filters, 3, 1, 1, bias=True)
        self.conv_x = nn.Conv2d(filters * 2, filters, 3, 1, 1, bias=True)
        self.conv_offset_2 = nn.Conv2d(filters * 2, filters, 3, 1, 1, bias=True)
        self.conv_dcnpack = DCN(filters, filters, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                                extra_offset_mask=True)
        self.lrule = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, ali, ref, offset_pre):
        tensor = self.lrule(self.conv(torch.cat([ali, ref], dim=1)))
        offset = self.lrule(self.conv_offset_1(tensor))
        x = self.lrule(self.conv_x(tensor))
        offset = self.lrule(self.conv_offset_2(torch.cat([offset_pre * 2, offset], dim=1)))
        output = self.lrule(self.conv_dcnpack([x, offset]))
        return output, offset


class Alignnet(nn.Module):
    def __init__(self, filters_in=64, nf=64, groups=8):
        super(Alignnet, self).__init__()
        self.GAMMA = 2.2

        self.L1_downsample_a = nn.Conv2d(filters_in, nf, 3, 2, 1, bias=True)
        self.L2_downsample_a = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.L3_downsample_a = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.L4_downsample_a = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        
        self.L1_downsample_r = nn.Conv2d(filters_in, nf, 3, 2, 1, bias=True)
        self.L2_downsample_r = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.L3_downsample_r = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.L4_downsample_r = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)

        self.L4_dcnpack = DCN_off_b(filters=nf, groups=groups)

        self.L2_dcnpack = DCN_off(filters=nf, groups=groups)

        self.L3_dcnpack = DCN_off(filters=nf, groups=groups)

        self.L1_dcnpack = DCN_off(filters=nf, groups=groups)

        self.L0_dcnpack = DCN_off(filters=nf, groups=groups)

        self.L3_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)

        self.L2_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)

        self.L1_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)

        self.L0_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, ali, ref):
        ali_L1 = self.lrelu(self.L1_downsample_a(ali))
        ali_L2 = self.lrelu(self.L2_downsample_a(ali_L1))
        ali_L3 = self.lrelu(self.L3_downsample_a(ali_L2))
        ali_L4 = self.lrelu(self.L4_downsample_a(ali_L3))
        
        ref_L1 = self.lrelu(self.L1_downsample_r(ref))
        ref_L2 = self.lrelu(self.L2_downsample_r(ref_L1))
        ref_L3 = self.lrelu(self.L3_downsample_r(ref_L2))
        ref_L4 = self.lrelu(self.L4_downsample_r(ref_L3))

        L4_aligned, L4_offset = self.L4_dcnpack(ali_L4, ref_L4)
        L4_aligned = F.interpolate(L4_aligned, scale_factor=2, mode='bilinear', align_corners=False)

        L4_offset = F.interpolate(L4_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L3_aligned, L3_offset = self.L3_dcnpack(ali_L3, ref_L3, L4_offset)
        L3_aligned = self.lrelu(self.L3_conv(torch.cat([L4_aligned, L3_aligned], dim=1)))
        L3_aligned = F.interpolate(L3_aligned, scale_factor=2, mode='bilinear', align_corners=False)

        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_aligned, L2_offset = self.L2_dcnpack(ali_L2, ref_L2, L3_offset)
        L2_aligned = self.lrelu(self.L2_conv(torch.cat([L3_aligned, L2_aligned], dim=1)))
        L2_aligned = F.interpolate(L2_aligned, scale_factor=2, mode='bilinear', align_corners=False)

        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_aligned, L1_offset = self.L1_dcnpack(ali_L1, ref_L1, L2_offset)
        L1_aligned = self.lrelu(self.L1_conv(torch.cat([L2_aligned, L1_aligned], dim=1)))
        L1_aligned = F.interpolate(L1_aligned, scale_factor=2, mode='bilinear', align_corners=False)

        L1_offset = F.interpolate(L1_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L0_aligned, L0_offset = self.L0_dcnpack(ali, ref, L1_offset)
        L0_aligned = self.lrelu(self.L0_conv(torch.cat([L1_aligned, L0_aligned], dim=1)))

        return L0_aligned, L0_offset


class make_res(nn.Module):
    def __init__(self, nFeat, kernel_size=3):
        super(make_res, self).__init__()
        self.conv1 = nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, padding=1, bias=True)
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, padding=1, bias=True)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out) + x
        return out


class Res_block(nn.Module):
    def __init__(self, nFeat, nDenselayer):
        super(Res_block, self).__init__()
        modules = []
        for i in range(nDenselayer):
            modules.append(make_res(nFeat))
        self.dense_layers = nn.Sequential(*modules)

    def forward(self, x):
        out = self.dense_layers(x)
        return out


class Mergenet(nn.Module):
    def __init__(self, nFeat=64, nDenselayer=3, filters_out=3):
        super(Mergenet, self).__init__()
        # fusion1
        self.conv_in1 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, padding=1, bias=True)
        self.conv_in3 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, padding=1, bias=True)
        self.res1 = Res_block(nFeat, nDenselayer)
        self.res3 = Res_block(nFeat, nDenselayer)

        # fusion2
        self.conv_stage2 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, padding=1, bias=True)
        self.res_satge2 = Res_block(nFeat, nDenselayer)
        self.conv_out = nn.Conv2d(nFeat, filters_out, kernel_size=3, padding=1, bias=True)

        # activation
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img1_aligned, img2, img3_aligned):
        img2_1 = self.relu(self.conv_in1(torch.cat([img1_aligned, img2], dim=1)))
        img2_3 = self.relu(self.conv_in3(torch.cat([img3_aligned, img2], dim=1)))

        img2_1 = self.res1(img2_1) + img2
        img2_3 = self.res3(img2_3) + img2

        output = self.relu(self.conv_stage2(torch.cat([img2_1, img2_3], dim=1)))
        output = self.res_satge2(output)
        output = self.sigmoid(self.conv_out(output))
        return output


# self_attention
class Self_sttention(nn.Module):
    def __init__(self, nFeat):
        super(Self_sttention, self).__init__()
        self.conv_img1 = nn.Conv2d(nFeat, nFeat, 3, 1, 1, bias=True)
        self.conv_img2 = nn.Conv2d(nFeat, nFeat, 3, 1, 1, bias=True)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        attention = self.relu(self.conv_img1(inputs))
        attention = self.sigmoid(self.conv_img2(attention))
        out = inputs * attention
        return out


class DHDR(nn.Module):
    def __init__(self, args):
        filters_in = args.filters_in
        nDenselayer = args.nDenselayer
        nFeat = args.nFeat
        filters_out = args.filters_out
        groups = args.groups
        self.args = args
        super(DHDR, self).__init__()
        self.GAMMA = 2.2

        self.conv_img1 = nn.Conv2d(filters_in, nFeat, 3, 1, 1, bias=True)
        self.conv_img2 = nn.Conv2d(filters_in, nFeat, 3, 1, 1, bias=True)
        self.conv_img3 = nn.Conv2d(filters_in, nFeat, 3, 1, 1, bias=True)

        self.Self_attention1 = Self_sttention(nFeat)
        self.Self_attention2 = Self_sttention(nFeat)
        self.Self_attention3 = Self_sttention(nFeat)

        self.alignnet1 = Alignnet(filters_in=nFeat, nf=nFeat, groups=groups)
        self.alignnet3 = Alignnet(filters_in=nFeat, nf=nFeat, groups=groups)
        self.mergenet = Mergenet(nFeat=nFeat, nDenselayer=nDenselayer, filters_out=filters_out)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, imgs):
        img1 = imgs[:, 0:6, :, :]
        img2 = imgs[:, 6:12, :, :]
        img3 = imgs[:, 12:18, :, :]

        img1 = self.lrelu(self.conv_img1(img1))
        img2 = self.lrelu(self.conv_img2(img2))
        img3 = self.lrelu(self.conv_img3(img3))

        img1 = self.Self_attention1(img1)
        img2 = self.Self_attention2(img2)
        img3 = self.Self_attention3(img3)

        img1_aligned, offset1 = self.alignnet1(img1, img2)
        img3_aligned, offset3 = self.alignnet3(img3, img2)
        out = self.mergenet(img1_aligned, img2, img3_aligned)
        return out

