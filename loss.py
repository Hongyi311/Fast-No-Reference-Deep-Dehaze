import torch
import torch.nn as nn
import torch.nn.functional as F


class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)

        return k


class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        b, c, h, w = org.shape

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        D_org_left = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_left - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = (D_left + D_right + D_up + D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E


class L_exp(nn.Module):

    def __init__(self, patch_size, mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
        return d


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class L_edge(nn.Module):
    def __init__(self, edge_weight=1):
        super(L_edge, self).__init__()
        self.edge_weight = edge_weight

    def forward(self, x):
        # Calculate gradient along horizontal and vertical directions
        enhance_mean = torch.mean(x, 1, keepdim=True)
        b, c, h, w = x.shape

        # Calculate smoothness loss using Laplacian operator
        laplacian_kernel = torch.tensor([[0.25, 0.5, 0.25], [0.5, -3, 0.5], [0.25, 0.5, 0.25]], dtype=torch.float32)
        x_padded = F.pad(enhance_mean, (1, 1, 1, 1), mode='reflect')
        laplacian_kernel = laplacian_kernel.to(x.device).unsqueeze(0).unsqueeze(0)

        sharpness_map = F.conv2d(x_padded, laplacian_kernel)

        edge_loss = self.edge_weight * sharpness_map.abs().mean()

        # Take the negative natural logarithm of the total_loss
        edge_loss = -torch.log(edge_loss)

        return edge_loss


class L_contrast(nn.Module):
    def __init__(self, contrast_weight=1):
        super(L_contrast, self).__init__()
        self.contrast_weight = contrast_weight

    def forward(self, x):
        # Calculate gradient along horizontal and vertical directions
        enhance_mean = torch.mean(x, 1, keepdim=True)

        gradient_x = torch.abs(enhance_mean[:, :, :, :-1] - enhance_mean[:, :, :, 1:])
        gradient_y = torch.abs(enhance_mean[:, :, :-1, :] - enhance_mean[:, :, 1:, :])
        contrast_loss = self.contrast_weight * (gradient_x.mean() + gradient_y.mean())

        contrast_loss = -torch.log(contrast_loss)

        return contrast_loss


# class L_est_dc(nn.Module):
#     def __init__(self, patch_size=8):
#         super(L_est_dc, self).__init__()
#         self.patch_size = patch_size
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#
#         # Reshape input into patches
#         patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
#         patches = patches.contiguous().view(b, c, -1, self.patch_size, self.patch_size)
#
#         # Calculate local minimum in each patch
#         min_values, _ = torch.min(patches, dim=4, keepdim=True)
#         min_values, _ = torch.min(min_values, dim=3, keepdim=True)
#         # Compute global minimum
#         dark_channel, _ = torch.min(min_values, dim=1, keepdim=True)
#
#         return dark_channel


class L_dc_partial(nn.Module):
    def __init__(self, patch_size=10, ):
        super(L_dc_partial, self).__init__()
        self.patch_size = patch_size

    def forward(self, x):
        b, c, h, w = x.shape

        # Reshape input into patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(b, c, -1, self.patch_size, self.patch_size)

        # Calculate local minimum in each patch
        min_values, _ = torch.min(patches, dim=4, keepdim=True)
        min_values, _ = torch.min(min_values, dim=3, keepdim=True)
        # Compute global minimum
        dark_channel_partial, _ = torch.min(min_values, dim=1, keepdim=True)
        threshold = 0.7
        mask = dark_channel_partial <= threshold
        dark_blocks = dark_channel_partial[mask.expand_as(dark_channel_partial)] - 0.1

        return dark_blocks


class L_bright(nn.Module):
    def __init__(self):
        super(L_bright, self).__init__()

    def forward(self, x, org):
        b, c, h, w = x.shape
        num_pixels = torch.numel(x)

        threshold = 0.7
        mask_x = x >= threshold
        white_pixel_x = x[mask_x]
        mask_org = org >= threshold
        white_pixel_org = org[mask_org]
        num_white_x = torch.numel(white_pixel_x)
        num_white_org = torch.numel(white_pixel_org)
        num_diff = (num_white_org - num_white_x)/ num_pixels

        return num_diff


def rgb_to_hsv(img):
    eps = 1e-8
    hue = torch.zeros((img.shape[0], img.shape[2], img.shape[3]), device=img.device)

    max_value, _ = img.max(1)
    min_value, _ = img.min(1)

    hue[img[:, 2] == max_value] = 4.0 + ((img[:, 0] - img[:, 1]) / (max_value - min_value + eps))[
        img[:, 2] == max_value]
    hue[img[:, 1] == max_value] = 2.0 + ((img[:, 2] - img[:, 0]) / (max_value - min_value + eps))[
        img[:, 1] == max_value]
    hue[img[:, 0] == max_value] = (0.0 + ((img[:, 1] - img[:, 2]) / (max_value - min_value + eps)))[
        img[:, 0] == max_value]

    hue[min_value == max_value] = 0.0
    hue = hue / 6.0

    saturation = (max_value - min_value) / (max_value + eps)
    saturation[max_value == 0] = 0.0

    value = max_value

    hue = hue.unsqueeze(1)
    saturation = saturation.unsqueeze(1)
    value = value.unsqueeze(1)
    hsv = torch.cat([hue, saturation, value], dim=1)
    return hsv


class L_hue(nn.Module):
    def __init__(self):
        super(L_hue, self).__init__()

    def forward(self, x):
        semi_inv = torch.max(x, 1 - x)
        hsv = rgb_to_hsv(x)
        hsv_inv = rgb_to_hsv(semi_inv)
        h = hsv[:, 0, :, :]
        h_inv = hsv_inv[:, 0, :, :]
        s = hsv[:, 1, :, :]
        v = hsv[:, 2, :, :]
        # hue_dis = 1 - torch.pow((h - h_inv), 2)
        hue_dis = (h - h_inv).abs().mean()
        hue_dis = -torch.log(hue_dis)

        return hue_dis


# class L_sat(nn.Module):
#     def __init__(self, patch_size=8):
#         super(L_sat, self).__init__()
#         self.patch_size = patch_size
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#
#         max_value = torch.max(x, dim=1, keepdim=True)[0]
#         min_value = torch.min(x, dim=1, keepdim=True)[0]
#         diff = 1 - min_value / (max_value + 0.000001)
#
#         # Reshape input into patches
#         patches = diff.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
#         patches = patches.contiguous().view(b, 1, -1, self.patch_size, self.patch_size)
#
#         # Calculate local minimum in each patch
#         min_val, _ = torch.max(patches, dim=4, keepdim=True)
#         min_val, _ = torch.max(min_val, dim=3, keepdim=True)
#         # Compute global minimum
#         sat = 1 - torch.max(min_val, dim=1, keepdim=True)[0]
#
#         return sat


class L_h_color(nn.Module):
    def __init__(self):
        super(L_h_color, self).__init__()

    def forward(self, org, enhance):
        hsv_org = rgb_to_hsv(org)
        hsv_en = rgb_to_hsv(enhance)
        h_org = hsv_org[:, 0, :, :]
        h_en = hsv_en[:, 0, :, :]
        hue_dif = torch.pow((h_org - h_en), 2)

        return hue_dif
