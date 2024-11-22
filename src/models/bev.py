import math
from operator import mul
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.models.resnet import ResNetLayer

# Roddick BEV Network

class BEVNET(pl.LightningModule):
    def __init__(self):
        super().__init__()

        t_res = 0.25 * reduce(mul, [1, 2])   # Map res 
        prior = [0.44679, 0.02407, 0.14491, 0.02994, 0.02086, 0.00477, 0.00156, 
                 0.00189, 0.00084, 0.00119, 0.00019, 0.00012, 0.00031, 0.00176]

        self.frontend = FPN()
        self.transformer = TransformerPyramid(resolution=t_res)
        self.topdown = TopdownNetwork()
        self.classifier = LinearClassifier(self.topdown.out_channels, 14)
        self.classifier.initialise(prior)

    def forward(self, image, calib, *args):
        feature_maps = self.frontend(image)
        bev_feats = self.transformer(feature_maps, calib)
        td_feats = self.topdown(bev_feats)
        logits = self.classifier(td_feats)
        return logits
    
class FPN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.in_planes = 64
        num_blocks = [3,4,6,3]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = ResNetLayer(64, 64, num_blocks[0], stride=1)
        self.layer2 = ResNetLayer(256, 128, num_blocks[1], stride=2)
        self.layer3 = ResNetLayer(512, 256, num_blocks[2], stride=2)
        self.layer4 = ResNetLayer(1024, 512, num_blocks[3], stride=2)

        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]))
        
    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        x = (x - self.mean.view(-1, 1, 1)) / self.std.view(-1, 1, 1)


        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))

        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        return p3, p4, p5, p6, p7
    
class TransformerPyramid(pl.LightningModule):
    def __init__(self, resolution, in_channels=256, channels=64, extents=[-25., 1., 25., 50.], ymin=-2, ymax=4, focal_length=630.):
        super().__init__()
        self.transformers = nn.ModuleList()
        for i in range(5):
            focal = focal_length / pow(2, i + 3)
            zmax = min(math.floor(focal * 2) * resolution, extents[3])
            zmin = math.floor(focal) * resolution if i < 4 else extents[1]
            subset_extents = [extents[0], zmin, extents[2], zmax]
            tfm = DenseTransformer(in_channels, channels, resolution, subset_extents, ymin, ymax, focal)
            self.transformers.append(tfm)
    

    def forward(self, feature_maps, calib):
        bev_feats = list()
        for i, fmap in enumerate(feature_maps):
            scale = 8 * 2 ** i
            calib_downsamp = calib.clone()
            calib_downsamp[:, :2] = calib[:, :2] / scale
            bev_feats.append(self.transformers[i](fmap, calib_downsamp))
        output = torch.cat(bev_feats[::-1], dim=-2)
        return output
    
class DenseTransformer(pl.LightningModule):

    def __init__(self, in_channels, channels, resolution, grid_extents, ymin, ymax, focal_length, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, channels, 1)
        self.bn = nn.GroupNorm(16, channels)
        self.resampler = Resampler(resolution, grid_extents)
        # Compute input height based on region of image covered by grid
        self.zmin, zmax = grid_extents[1], grid_extents[3]
        self.in_height = math.ceil(focal_length * (ymax - ymin) / self.zmin)
        self.ymid = (ymin + ymax) / 2
        # Compute number of output cells required
        self.out_depth = math.ceil((zmax - self.zmin) / resolution)
        # Dense layer which maps UV features to UZ
        self.fc = nn.Conv1d(channels * self.in_height, channels * self.out_depth, 1, groups=groups)
        self.out_channels = channels
    
    def forward(self, features, calib, *args):
        # Crop feature maps to a fixed input height
        features = torch.stack([self._crop_feature_map(fmap, cal) for fmap, cal in zip(features, calib)])
        # Reduce feature dimension to minimize memory usage
        features = F.relu(self.bn(self.conv(features)))
        # Flatten height and channel dimensions
        B, C, _, W = features.shape
        flat_feats = features.flatten(1, 2)
        bev_feats = self.fc(flat_feats).view(B, C, -1, W)
        # Resample to orthographic grid
        return self.resampler(bev_feats, calib)

    def _crop_feature_map(self, fmap, calib):
        # Compute upper and lower bounds of visible region
        focal_length, img_offset = calib[1, 1:]
        vmid = self.ymid * focal_length / self.zmin + img_offset
        vmin = math.floor(vmid - self.in_height / 2)
        vmax = math.floor(vmid + self.in_height / 2)
        # Pad or crop input tensor to match dimensions
        return F.pad(fmap, [0, 0, -vmin, vmax - fmap.shape[-2]])
    
class Resampler(pl.LightningModule):
    def __init__(self, resolution, extents):
        super().__init__()
        self.near = extents[1]
        self.far = extents[3]
        self.grid = _make_grid(resolution, extents)


    def forward(self, features, calib):
        self.grid = self.grid.to(features)
        calib = calib[:, [0, 2]][..., [0, 2]].view(-1, 1, 1, 2, 2)
        cam_coords = torch.matmul(calib, self.grid.unsqueeze(-1)).squeeze(-1)
        ucoords = cam_coords[..., 0] / cam_coords[..., 1]
        ucoords = ucoords / features.size(-1) * 2 - 1
        zcoords = (cam_coords[..., 1]-self.near) / (self.far-self.near) * 2 - 1
        grid_coords = torch.stack([ucoords, zcoords], -1).clamp(-1.1, 1.1)
        return F.grid_sample(features, grid_coords, align_corners=False) # False or True? To remove warning

class TopdownNetwork(nn.Sequential):
    def __init__(self, in_channels=64, channels=128, layers=[4, 4], strides=[1, 2], blocktype='bottleneck'):
        modules = list()
        self.downsample = 1
        for nblocks, stride in zip(layers, strides):
            module = ResNetLayer(in_channels, channels, nblocks, 1/stride, blocktype=blocktype)
            modules.append(module)
            in_channels = module.out_channels
            channels = channels // 2
            self.downsample *= stride
        self.out_channels = in_channels
        super().__init__(*modules)

class TopdownNetwork3(nn.Sequential):
    def __init__(self, in_channels=64, channels=128, layers=[4, 4], strides=[1, 2], blocktype='bottleneck'):
        modules = list()
        self.downsample = 1
        for nblocks, stride in zip(layers, strides):
            module = ResNetLayer(in_channels, channels, nblocks, 1/stride, blocktype=blocktype)
            modules.append(module)
            in_channels = module.out_channels
            channels = channels // 2
            self.downsample *= stride
        self.out_channels = in_channels
        super().__init__(*modules)

class LinearClassifier(nn.Conv2d):
    def __init__(self, in_channels, num_class):
        super().__init__(in_channels, num_class, 1)
    
    def initialise(self, prior):
        prior = torch.tensor(prior)
        self.weight.data.zero_()
        self.bias.data.copy_(torch.log(prior / (1 - prior)))

def _make_grid(resolution, extents):
    x1, z1, x2, z2 = extents
    zz, xx = torch.meshgrid(torch.arange(z1, z2, resolution), torch.arange(x1, x2, resolution))
    return torch.stack([xx, zz], dim=-1)


# BEVNET for expanding into 3 channel satellite images
class BEVNET3(pl.LightningModule):
    def __init__(self):
        super().__init__()

        t_res = 0.25 * reduce(mul, [1, 2])   # Map res 
        prior = [0.44679, 0.02407, 0.14491, 0.02994, 0.02086, 0.00477, 0.00156, 0.00189, 0.00084, 0.00119, 0.00019, 0.00012, 0.00031, 0.00176]

        self.frontend = FPN()
        self.transformer = TransformerPyramid(resolution=t_res)
        self.topdown = TopdownNetwork()
        self.classifier = LinearClassifier(self.topdown.out_channels, 14)
        self.classifier.initialise(prior)

    def forward(self, image, calib, *args):
        feature_maps = self.frontend(image)
        bev_feats = self.transformer(feature_maps, calib)
        td_feats = self.topdown(bev_feats)
        logits = self.classifier(td_feats)
        return logits