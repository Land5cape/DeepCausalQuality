# coding:utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ot.lp import wasserstein_1d
from torchvision import models
from torchvision.models import VGG16_Weights

from tools.utils import downsample


class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]

        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer('filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, input):
        input = input ** 2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out + 1e-12).sqrt()

def causal_intervene_channel(x, y, P=2, num_sigma_samples=10, max_intensity=0.1, step=10):
    # x.shape: # B, C, H//win * W//win, win * win
    # return: channel_causal_intensity: B, C, 1

    B, C, patch_num_channel, patch_size = x.shape

    index = np.arange(0, patch_size, 1)
    # print(index)
    all_samples = torch.from_numpy(index).repeat([B, C, patch_num_channel, 1]).to(x.device)

    all_samples = all_samples.permute(3, 0, 1, 2)
    x_p = x.permute(3, 0, 1, 2)
    y_p = y.permute(3, 0, 1, 2)
    ot = wasserstein_1d(all_samples, all_samples, x_p, y_p, p=P).sum(dim=2)

    # print(ot_1)
    # print(x.shape)  # [1, 3, 4096, 16]
    # print(ot_1.shape) # [1, 3, 4096]

    # 为每个B, C生成多个不同的sigma值（num_sigma_samples个）

    # intensity_samples = torch.rand(num_sigma_samples, 1, C, 1, 1, device=x.device)
    # intensity_samples = intensity_samples.repeat([1, B, 1, patch_num_channel, patch_size]).to(x.device)

    # intensity_samples = torch.rand(num_sigma_samples, 1, C, device=x.device) # * 0.1
    # intensity_samples = intensity_samples.repeat([1, B, 1]).to(x.device)

    intensity_values = torch.arange(0.0000, max_intensity, max_intensity / step,
                                    device=x.device)  # 结果：[0.01, 0.02, ..., 0.1]
    intensity_samples = intensity_values.view(-1, 1, 1).repeat(1, B, C).to(x.device)

    # print(intensity_samples.shape)
    # print(intensity_samples)

    # 用来存储每个B, C上的最大扰动强度和对应的sigma
    best_intensity = torch.zeros(B, C, device=x.device)

    x_original = x.clone().to(x.device)
    y_original = y.clone().to(x.device)

    ####################################################################################################################
    # Generate disturbance for all intensity samples
    _mu = torch.zeros_like(x).unsqueeze(0).repeat(intensity_samples.size(0), 1, 1, 1, 1).to(x.device)
    _intensity = (intensity_samples.view(intensity_samples.size(0), B, C, 1, 1)
                  .repeat(1, 1, 1, patch_num_channel, patch_size))
    disturbance = (torch.normal(_mu, 1).to(x.device) * torch.mean((x + y) / 2, dim=[1, 2, 3], keepdim=True)
                     .unsqueeze(0) * _intensity)

    # Add disturbance
    x_disturbed = x_original.unsqueeze(0) + disturbance
    y_disturbed = y_original.unsqueeze(0) + disturbance

    _index = np.arange(0, patch_size, 1)
    _all_samples = torch.from_numpy(_index).repeat([B, C, patch_num_channel, 1]).to(x.device)
    _all_samples = _all_samples.unsqueeze(0).repeat(intensity_samples.size(0), 1, 1, 1, 1).permute(4, 0, 1, 2, 3)
    x_p_disturbed = x_disturbed.permute(4, 0, 1, 2, 3)
    y_p_disturbed = y_disturbed.permute(4, 0, 1, 2, 3)

    # Calculate Wasserstein distance
    ot_disturbed = wasserstein_1d(_all_samples, _all_samples, x_p_disturbed, y_p_disturbed, P).sum(dim=3)

    # Calculate mask
    mask = (torch.abs(ot.unsqueeze(0) - ot_disturbed) < torch.abs(torch.sum((x - y) / 2, dim=[1, 2, 3]))
              .unsqueeze(0) * 1e-4)

    # Update best intensity
    best_intensity_expanded = best_intensity.view(1, B, C).repeat(intensity_samples.size(0), 1, 1)

    intensity_samples_expanded = intensity_samples.view(intensity_samples.size(0), B, C)

    best_intensity_expanded[mask] = torch.max(best_intensity_expanded[mask], intensity_samples_expanded[mask])
    best_intensity = best_intensity_expanded.max(dim=0)[0]

    # print(best_intensity.shape)

    channel_causal_intensity = max_intensity - best_intensity
    # print('channel_causal_intensity', channel_causal_intensity)
    return channel_causal_intensity



def mv_distance_causal(X, Y, win=4, P=2, pdf_mode=0):
    B, C, H, W = X.shape
    X_sum = X.sum(dim=[1, 2, 3])
    Y_sum = Y.sum(dim=[1, 2, 3])

    X_patch = torch.reshape(X, [B, C, H // win, win, W // win, win])
    Y_patch = torch.reshape(Y, [B, C, H // win, win, W // win, win])

    # B, C, H // win, W // win, win, win
    X_patch = X_patch.permute(0, 1, 2, 4, 3, 5)
    Y_patch = Y_patch.permute(0, 1, 2, 4, 3, 5)

    X_CD = torch.reshape(X_patch, [B, C, -1, win * win])
    Y_CD = torch.reshape(Y_patch, [B, C, -1, win * win])

    # pdf_mode = -1  #TODO test
    if pdf_mode == -1:
        X_pdf = X_CD
        Y_pdf = Y_CD
    elif pdf_mode == 0:
        X_CD_pdf = X_CD / (X_sum.view(B, 1, 1, 1) + 1e-6)
        Y_CD_pdf = Y_CD / (Y_sum.view(B, 1, 1, 1) + 1e-6)
        X_pdf = X_CD * X_CD_pdf
        Y_pdf = Y_CD * Y_CD_pdf
    else:
        X_pdf = F.softmax(X_CD, dim=2)
        Y_pdf = F.softmax(Y_CD, dim=2)

    # 获得通道维度强度图  B, C
    max_intensity = 0.1
    step = 10
    channel_causal_intensity = causal_intervene_channel(X_pdf, Y_pdf, max_intensity=max_intensity, step=step)

    # causal_mask
    mask = channel_causal_intensity != max_intensity
    channel_causal_intensity[mask] = 0

    # ablation
    # mask = channel_causal_intensity == max_intensity
    # channel_causal_intensity[mask] = 0

    # 根据相应强度计算距离
    B, C, patch_num_channel, patch_size = X_pdf.shape
    index = np.arange(0, patch_size, 1)

    all_samples = torch.from_numpy(index).repeat([B, C, patch_num_channel, 1]).to(X.device)
    all_samples = all_samples.permute(3, 0, 1, 2)

    # wd
    x_p = X_pdf.permute(3, 0, 1, 2) * channel_causal_intensity.view(1, B, C, 1) / max_intensity
    y_p = Y_pdf.permute(3, 0, 1, 2) * channel_causal_intensity.view(1, B, C, 1) / max_intensity
    ot = wasserstein_1d(all_samples, all_samples, x_p, y_p, P)  # B, C, patch_num_channel
    tmp_ot = torch.reshape(ot, [B, C, H // win, W // win])

    # md
    X_pdf = X_pdf * channel_causal_intensity.view(B, C, 1, 1) / max_intensity
    Y_pdf = Y_pdf * channel_causal_intensity.view(B, C, 1, 1) / max_intensity
    md = torch.norm(X_pdf - Y_pdf, p=2, dim=3, keepdim=True)
    tmp_md = torch.reshape(md, [B, C, H // win, W // win])

    # L2
    X_CD = X_CD * channel_causal_intensity.view(B, C, 1, 1) / max_intensity
    Y_CD = Y_CD * channel_causal_intensity.view(B, C, 1, 1) / max_intensity
    L2 = ((X_CD - Y_CD) ** 2).sum(dim=3, keepdim=True)
    tmp_L2 = torch.reshape(L2, [B, C, H // win, W // win])

    # final_tmp = tmp_md  # LIVE CSIQ
    # final_tmp = tmp_L2  # TID2008
    # final_tmp = tmp_ot
    final_tmp = tmp_ot + tmp_md  # TID2013 KADID PIPAL
    # final_tmp = tmp_ot + tmp_L2

    final = final_tmp.sum(dim=[1, 2, 3])

    # print(tmp_ot.sum(), tmp_md.sum(), tmp_L2.sum())
    return final, patch_num_channel, final_tmp


class DeepCausalQualityVGG(torch.nn.Module):

    def __init__(self, channels=3, pretrained=True, device=torch.device('cpu')):
        assert channels == 3
        super(DeepCausalQualityVGG, self).__init__()

        self.lower_better = True
        self.window = 8

        self.device = device

        # vgg_pretrained_features = models.vgg16(pretrained=True).features
        vgg_pretrained_features = models.vgg16(weights=VGG16_Weights.DEFAULT).features
        # weigth_init(vgg_pretrained_features)

        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()

        # Rewrite the output layer of every block in the VGG network: maxpool->l2pool
        for x in range(0, 4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

        self.chns = [3, 64, 128, 256, 512, 512]

    def forward_once(self, x):
        h = x
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def foward_distance(self, feats0, feats1, window):
        distances = list()
        nums = list()

        for k in range(len(self.chns)):
            row_padding = round(feats0[k].size(2) / window) * window - feats0[k].size(2)
            column_padding = round(feats0[k].size(3) / window) * window - feats0[k].size(3)

            pad = nn.ZeroPad2d((column_padding, 0, 0, row_padding))
            feats0_k = pad(feats0[k])
            feats1_k = pad(feats1[k])

            # tmp, num = mv_distance_sample(feats0_k, feats1_k, win=window)
            # tmp, num, tmp_md = mv_distance_patch_causal(feats0_k, feats1_k, win=window)
            tmp, num, tmp_md = mv_distance_causal(feats0_k, feats1_k, win=window, pdf_mode=0)
            if not tmp.size():
                tmp = tmp.unsqueeze(0)

            distances.append(tmp)
            nums.append(num)

        distances = torch.stack(distances).transpose(1, 0).to(self.device)
        distances = torch.mean(distances, dim=1)  # * sum(nums)
        # distances = torch.sum(distances, dim=1)
        return distances

    def forward(self, x, y, as_loss=True, resize=True):
        assert x.shape == y.shape

        if resize:
            x, y, _, _ = downsample(x, y)
        if as_loss:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y)

        distance = self.foward_distance(feats0, feats1, self.window)

        return distance

class DeepCausalQualityEFF(torch.nn.Module):

    def __init__(self, channels=3, device=torch.device('cpu')):
        assert channels == 3
        super(DeepCausalQualityEFF, self).__init__()

        self.window = 4
        self.device = device
        self.lower_better = True

        effb7_pretrained_features = models.efficientnet_b7(pretrained=True).features
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        self.stage6 = torch.nn.Sequential()
        self.stage7 = torch.nn.Sequential()

        for x in range(0, 2):
            self.stage1.add_module(str(x), effb7_pretrained_features[x])
        for x in range(2, 3):
            self.stage2.add_module(str(x), effb7_pretrained_features[x])
        for x in range(3, 4):
            self.stage3.add_module(str(x), effb7_pretrained_features[x])
        for x in range(4, 5):
            self.stage4.add_module(str(x), effb7_pretrained_features[x])
        for x in range(5, 6):
            self.stage5.add_module(str(x), effb7_pretrained_features[x])
        for x in range(6, 7):
            self.stage6.add_module(str(x), effb7_pretrained_features[x])
        for x in range(7, 8):
            self.stage7.add_module(str(x), effb7_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

        self.chns = [3, 16, 24, 40, 80, 112, 192, 320]

    def forward_once(self, x):
        h = x
        h = self.stage1(h)
        h_1 = h
        h = self.stage2(h)
        h_2 = h
        h = self.stage3(h)
        h_3 = h
        h = self.stage4(h)
        h_4 = h
        h = self.stage5(h)
        h_5 = h
        h = self.stage6(h)
        h_6 = h
        h = self.stage7(h)
        h_7 = h

        return [x, h_1, h_2, h_3, h_4, h_5, h_6, h_7]

    def foward_distance(self, feats0, feats1, window):
        distances = list()
        nums = list()

        for k in range(len(self.chns)):
            row_padding = round(feats0[k].size(2) / window) * window - feats0[k].size(2)
            column_padding = round(feats0[k].size(3) / window) * window - feats0[k].size(3)

            pad = nn.ZeroPad2d((column_padding, 0, 0, row_padding))
            feats0_k = pad(feats0[k])
            feats1_k = pad(feats1[k])

            # tmp, num = mv_distance_sample(feats0_k, feats1_k, win=window)
            # tmp, num, tmp_md = mv_distance_patch_causal(feats0_k, feats1_k, win=window)
            tmp, num, tmp_md = mv_distance_causal(feats0_k, feats1_k, win=window, pdf_mode=1)
            if not tmp.size():
                tmp = tmp.unsqueeze(0)

            distances.append(tmp)
            nums.append(num)

        distances = torch.stack(distances).transpose(1, 0).to(self.device)
        distances = torch.mean(distances, dim=1)  # * sum(nums)
        # distances = torch.sum(distances, dim=1)
        return distances


    def forward(self, x, y, as_loss=True, resize=True):
        assert x.shape == y.shape
        if resize:
            x, y, _ , _ = downsample(x, y)
        if as_loss:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y)

        distance = self.foward_distance(feats0, feats1, self.window)
        return distance

class DeepCausalQualityRES(torch.nn.Module):

    def __init__(self, device=torch.device('cpu')):
        super(DeepCausalQualityRES, self).__init__()

        self.window = 4
        self.device = device
        self.lower_better = True

        resnet_pretrained_features = models.resnet50(pretrained=True)
        # print(resnet_pretrained_features)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()

        pre_stage = torch.nn.Sequential(
            resnet_pretrained_features.conv1,
            resnet_pretrained_features.bn1,
            resnet_pretrained_features.relu,
            resnet_pretrained_features.maxpool
        )
        self.stage1.add_module(str(0), pre_stage)
        self.stage1.add_module(str(1), resnet_pretrained_features.layer1)
        self.stage2.add_module(str(2), resnet_pretrained_features.layer2)
        self.stage3.add_module(str(3), resnet_pretrained_features.layer3)
        self.stage4.add_module(str(4), resnet_pretrained_features.layer4)

        for param in self.parameters():
            param.requires_grad = False

        self.chns = [3, 256, 512, 1024, 2048]

    def forward_once(self, x):
        h = x
        h = self.stage1(h)
        h_1 = h
        h = self.stage2(h)
        h_2 = h
        h = self.stage3(h)
        h_3 = h
        h = self.stage4(h)
        h_4 = h

        return [x, h_1, h_2, h_3, h_4]

    def foward_distance(self, feats0, feats1, window):
        distances = list()
        nums = list()

        for k in range(len(self.chns)):
            row_padding = round(feats0[k].size(2) / window) * window - feats0[k].size(2)
            column_padding = round(feats0[k].size(3) / window) * window - feats0[k].size(3)

            pad = nn.ZeroPad2d((column_padding, 0, 0, row_padding))
            feats0_k = pad(feats0[k])
            feats1_k = pad(feats1[k])

            # tmp, num = mv_distance_sample(feats0_k, feats1_k, win=window)
            # tmp, num, tmp_md = mv_distance_patch_causal(feats0_k, feats1_k, win=window)
            tmp, num, tmp_md = mv_distance_causal(feats0_k, feats1_k, win=window, pdf_mode=1)
            if not tmp.size():
                tmp = tmp.unsqueeze(0)

            distances.append(tmp)
            nums.append(num)

        distances = torch.stack(distances).transpose(1, 0).to(self.device)
        distances = torch.mean(distances, dim=1)  # * sum(nums)
        # distances = torch.sum(distances, dim=1)
        return distances

    def forward(self, x, y, as_loss=True, resize=True):
        assert x.shape == y.shape
        if resize:
            x, y, _, _ = downsample(x, y)
        if as_loss:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y)

        distance = self.foward_distance(feats0, feats1, self.window)
        return distance

# ==================================================
#  Test Part

if __name__ == '__main__':
    from PIL import Image
    import argparse
    from torchvision import transforms

    def prepare_image_dcq(image):
        H, W = image.size
        if max(H, W) > 288:
            image = transforms.functional.resize(image, [288, 288])
        image = transforms.ToTensor()(image).unsqueeze(0)
        return image

    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='../images/1600.png')
    parser.add_argument('--dist', type=str, default='../images/1600.BLUR.5.png')
    args = parser.parse_args()

    device = torch.device('cpu')

    ref = prepare_image_dcq(Image.open(args.ref).convert("RGB")).to(device)
    dist = prepare_image_dcq(Image.open(args.dist).convert("RGB")).to(device)

    model = DeepCausalQualityVGG().to(device)
    # print(model)
    score = model(ref, dist)
    print(score)
