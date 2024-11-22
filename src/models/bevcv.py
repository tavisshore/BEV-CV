from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import torchvision.transforms as T
from torchvision.transforms.functional import rotate, center_crop
import pytorch_lightning as pl
from pytorch_metric_learning.losses import NTXentLoss
import math
from PIL import Image
import scipy.spatial as ss

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import pandas as pd
from torchvision.models import convnext_small, ConvNeXt_Small_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from typing import Any, Union
from pathlib import Path
import io
import lmdb
import pickle
from PIL import Image, ImageFile

import math
from operator import mul
from functools import reduce
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile



ImageFile.LOAD_TRUNCATED_IMAGES = True


from yacs.config import CfgNode as CN


def get_cfg_defaults():
    # Default configs for BEV-CV 
    _C = CN()
    _C.ex = 0
    _C.epochs = 150
    _C.batch_size = 32
    _C.config = 'cvusa/sub_exp_1'
    _C.train_acc_every = 5
    _C.workers = 4
    _C.learning_rate = 0.0001

    _C.align = True # using corr orientations to rotate here
    _C.train_true_align = False # Train with GT yaw
    _C.test_true_align = False # Test with GT yaw
    _C.train_north = False
    _C.test_north = False
    _C.train_align = True
    _C.test_align = True
    _C.multi_kd = False


    _C.rotations = 0
    _C.aug = 0
    _C.dims = 512
    _C.resume_training = False
    _C.log_name = 'default_log'
    _C.top_ks = ['1', '5', '10', '1%']
    _C.test_aug = 0
    _C.seed = 42

    # Data - {self.cfg.dataset}_train_{self.cfg.fov}_{self.cfg.zoom}_{self.cfg.resized}.beton
    _C.dataset = 'CVUSA'
    _C.fov = 90
    _C.zoom = 19
    _C.resized = 256

    _C.drop_zeros = False
    _C.drop_zeros_threshold = 2500
    _C.rot_step = 1
    _C.double_acc = False

    path = Path(__file__).parent.parent
    drive = path.parts[1]
    # print(f'path: {path}, drive: {drive}')
    if drive == 'mnt':
        cv_path = '/mnt/fast/nobackup/scratch4weeks/ts00987/vol/research/deep_localisation/CV_data'
        save_path = '/mnt/fast/nobackup/users/ts00987/vol/research/deep_localisation/bev-cv-polish/trained_models/'
    elif drive == 'vol':
        cv_path = '/vol/research/NOBACKUP/CVSSP/scratch_4weeks/tav/CV_data'
        # cv_path = '/scratch/CVUSA/ffcv/'
        save_path = '/vol/research/deep_localisation/bev-cv-polish/trained_models/'
    else: # Make MBP path
        cv_path = '/scratch/datasets/'
        save_path = 'weights/'

    _C.path = str(path)
    _C.cv_data = cv_path
    _C.save_path = save_path

    return _C.clone()






def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    if stride < 1:
        stride = int(round(1 / stride))
        kernel_size = stride + 2
        padding = int((dilation * (kernel_size - 1) - stride + 1) / 2)
        return nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size, stride, padding, 
            output_padding=0, dilation=dilation, bias=False)
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=int(stride), dilation=dilation, padding=dilation, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    if int(1 / stride) > 1:
        stride = int(1 / stride)
        return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=stride, stride=stride,bias=False)
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=int(stride), bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.GroupNorm(16, planes)

        self.conv2 = conv3x3(planes, planes, 1, dilation)
        self.bn2 = nn.GroupNorm(16, planes)

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride), nn.GroupNorm(16, planes))
        else:
            self.downsample = None


    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out, inplace=True)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.GroupNorm(16, planes)
        self.conv2 = conv3x3(planes, planes, stride, dilation)
        self.bn2 = nn.GroupNorm(16, planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.GroupNorm(16, planes * self.expansion)

        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * self.expansion, stride), 
                nn.GroupNorm(16, planes * self.expansion))
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
 
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out


class ResNetLayer(nn.Sequential):
    def __init__(self, in_channels, channels, num_blocks, stride=1, dilation=1, blocktype='bottleneck'):
        if blocktype == 'basic':
            block = BasicBlock
        elif blocktype == 'bottleneck':
            block = Bottleneck
        else:
            raise Exception("Unknown residual block type: " + str(blocktype))
        
        layers = [block(in_channels, channels, stride, dilation)]
        for _ in range(1, num_blocks):
            layers.append(block(channels * block.expansion, channels, 1, dilation))

        self.in_channels = in_channels
        self.out_channels = channels * block.expansion

        super(ResNetLayer, self).__init__(*layers)


def _ascii_encode(data: str) -> bytes:
    return data.encode("ascii")

def _pickle_encode(data: Any, protocol: int) -> bytes:
    return pickle.dumps(data, protocol=protocol)

def _pickle_decode(data: bytes) -> Any:
    return pickle.loads(data)


class Database(object):
    _database = None
    _protocol = None
    _length = None
    _has_fetched = False

    def __init__(
        self,
        path: Union[str, Path],
        readahead: bool = False,
    ):
        """
        Base class for LMDB-backed databases.

        :param path: Path to the database.
        :param readahead: Enables the filesystem readahead mechanism. Useful only if your database fits in RAM.
        """
        if not isinstance(path, str):
            path = str(path)

        self.path = path
        self.readahead = readahead

    @property
    def database(self):
        if self._database is None:
            self._database = lmdb.open(
                path=self.path,
                readonly=True,
                readahead=self.readahead,
                max_spare_txns=256,
                lock=False,
            )
        return self._database

    @database.deleter
    def database(self):
        if self._database is not None:
            self._database.close()
            self._database = None

    @property
    def protocol(self):
        """
        Read the pickle protocol contained in the database.

        :return: The pickle protocol.
        """
        if self._protocol is None:
            self._protocol = self._get(
                key="protocol",
                fencode=_ascii_encode,
                fdecode=_pickle_decode,
            )
        return self._protocol

    @property
    def keys(self):
        """
        Read the keys contained in the database.

        :return: The set of available keys.
        """
        protocol = self.protocol
        keys = self._get(
            key="keys",
            fencode=lambda key: _pickle_encode(key, protocol=protocol),
            fdecode=_pickle_decode,
        )
        return keys

    def __len__(self):
        """
        Returns the number of keys available in the database.

        :return: The number of keys.
        """
        if self._length is None:
            self._length = len(self.keys)
        return self._length

    def __getitem__(self, item):
        """
        Retrieves an item or a list of items from the database.

        :param item: A key or a list of keys.
        :return: A value or a list of values.
        """
        self._has_fetched = True
        if not isinstance(item, list):
            item = self._get(
                key=item,
                fencode=self._fencode,
                fdecode=self._fdecode,
            )
        else:
            item = self._gets(
                keys=item,
                fencodes=self._fencodes,
                fdecodes=self._fdecodes,
            )
        return item

    def _get(self, key, fencode, fdecode):
        """
        Instantiates a transaction and its associated cursor to fetch an item.

        :param key: A key.
        :param fencode:
        :param fdecode:
        :return:
        """
        with self.database.begin() as txn:
            with txn.cursor() as cursor:
                key = fencode(key)
                value = cursor.get(key)
                value = fdecode(value)
        self._keep_database()
        return value

    def _gets(self, keys, fencodes, fdecodes):
        """
        Instantiates a transaction and its associated cursor to fetch a list of items.

        :param keys: A list of keys.
        :param fencodes:
        :param fdecodes:
        :return:
        """
        with self.database.begin() as txn:
            with txn.cursor() as cursor:
                keys = fencodes(keys)
                _, values = list(zip(*cursor.getmulti(keys)))
                values = fdecodes(values)
        self._keep_database()
        return values

    def _fencode(self, key: Any) -> bytes:
        """
        Converts a key into a byte key.

        :param key: A key.
        :return: A byte key.
        """
        return _pickle_encode(data=key, protocol=self.protocol)

    def _fencodes(self, keys: list[Any]) -> list[bytes]:
        """
        Converts keys into byte keys.

        :param keys: A list of keys.
        :return: A list of byte keys.
        """
        return [self._fencode(key=key) for key in keys]

    def _fdecode(self, value: bytes) -> Any:
        """
        Converts a byte value back into a value.

        :param value: A byte value.
        :return: A value
        """
        return _pickle_decode(data=value)

    def _fdecodes(self, values: list[bytes]) -> list[Any]:
        """
        Converts bytes values back into values.

        :param values: A list of byte values.
        :return: A list of values.
        """
        return [self._fdecode(value=value) for value in values]

    def _keep_database(self):
        """
        Checks if the database must be deleted.

        :return:
        """
        if not self._has_fetched:
            del self.database

    def __iter__(self):
        """
        Provides an iterator over the keys when iterating over the database.

        :return: An iterator on the keys.
        """
        return iter(self.keys)

    def __del__(self):
        """
        Closes the database properly.
        """
        del self.database


class ImageDatabase(Database):
    def _fdecode(self, value: bytes):
        value = io.BytesIO(value)
        image = Image.open(value)
        return image




def accuracy(pov_embs, map_embs, workers=4, length=None, config=None):
    if map_embs is None or pov_embs is None: return None

    with torch.no_grad():
        k_tops = [1, 5, 10, math.ceil(0.01 * length)]
        # print(f'k_tops: {k_tops}')
        accs = []

        pov_embeddings = pov_embs.cpu().numpy()
        map_embeddings = map_embs.cpu().numpy()

        ##### KDTree #####
        t = ss.KDTree(data=map_embeddings)

        for k_value in k_tops:
            idx, count = 0, 0
            _, nn_idx = t.query(pov_embeddings, k=k_value, workers=workers)

            it = iter(nn_idx)
            for emb in it:
                if config.double_acc:
                    emb = np.concatenate((emb, next(it)))
                    idxes = np.array([idx, idx+1])
                    idx += 2
                else:
                    idxes = np.array([idx])
                    idx += 1
                mask = np.isin(idxes, emb)
                if mask.any(): 
                    count += 1
                    # print(f'idx: {idx}, emb: {emb}, True')
            acc = round((count / length) * 100, 6)
            accs.append(acc)
    pov_embeddings = None
    map_embeddings = None 
    t = None
    return accs



# Roddick BEV Network
class ProjectionModule(pl.LightningModule):
    def __init__(self, config, scale=1):
        super(ProjectionModule, self).__init__()
        self.config=config
        self.pov_proj = nn.Sequential(
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU(),
            nn.Linear(512, self.config.dims),
        )
        self.map_proj = nn.Sequential(
            nn.BatchNorm1d(num_features=768),
            nn.LeakyReLU(),
            nn.Linear(512, self.config.dims),
        )
        self.scale = scale

    def forward(self, pov, map):
        pov_out = self.pov_proj(pov)
        map_out = self.map_proj(map)
        return pov_out, map_out


class ShapeStandardiser(pl.LightningModule):
    def     __init__(self):
        super(ShapeStandardiser, self).__init__()
        self.mp = nn.Sequential(
            nn.MaxPool2d(kernel_size=(7, 7), stride=None, padding=0),
            nn.Flatten(1, -1),
        )
        self.reduce = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU()
        )
        self.scale = 1

    def standard_shape(self, x):
        shape = x.shape
        if len(shape) > 2:
            if shape[2] == 7:
                x = self.mp(x)
        if shape[1] == 2048:
            x = self.reduce(x)

        x = x.view(-1, 512)
        x_in = F.normalize(x, p=2, dim=1) * self.scale
        return x_in
    
    def forward(self, pov=None, map=None):
        if pov == None and map==None:
            raise ValueError('Both pov and map cannot be None')
        if pov == None:
            map_out = self.standard_shape(map)
            return map_out
        if map == None:
            pov_out = self.standard_shape(pov)
            return pov_out
        if pov != None and map != None:
            pov_out = self.standard_shape(pov)
            map_out = self.standard_shape(map)
            return pov_out, map_out


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


class USADataset(Dataset):
    def __init__(self, data_dir="/scratch/datasets/CVUSA/files/", geometric_aug='strong', sematic_aug='strong', mode='train', is_polar=True, is_mutual=True, fov=90):
        self.data_dir = data_dir
        self.fov = fov
        STREET_IMG_WIDTH = 256
        STREET_IMG_HEIGHT = 256
        SATELLITE_IMG_WIDTH = 256
        SATELLITE_IMG_HEIGHT = 256

        self.is_polar = is_polar
        self.mode = mode
        self.is_mutual = is_mutual

        transforms_street = [transforms.Resize((STREET_IMG_HEIGHT, STREET_IMG_WIDTH))]
        transforms_sat = [transforms.Resize((SATELLITE_IMG_HEIGHT, SATELLITE_IMG_WIDTH))]

        if sematic_aug == 'strong' or sematic_aug == 'same':
            transforms_sat.append(transforms.ColorJitter(0.3, 0.3, 0.3))
            transforms_street.append(transforms.ColorJitter(0.3, 0.3, 0.3))

            transforms_sat.append(transforms.RandomGrayscale(p=0.2))
            transforms_street.append(transforms.RandomGrayscale(p=0.2))

            try:
                transforms_sat.append(transforms.RandomPosterize(p=0.2, bits=4))
                transforms_street.append(transforms.RandomPosterize(p=0.2, bits=4))
            except:
                transforms_sat.append(RandomPosterize(p=0.2, bits=4))
                transforms_street.append(RandomPosterize(p=0.2, bits=4))

            transforms_sat.append(transforms.GaussianBlur(kernel_size=(1, 5), sigma=(0.1, 5)))
            transforms_street.append(transforms.GaussianBlur(kernel_size=(1, 5), sigma=(0.1, 5)))

        elif sematic_aug == 'weak':
            transforms_sat.append(transforms.ColorJitter(0.1, 0.1, 0.1))
            transforms_street.append(transforms.ColorJitter(0.1, 0.1, 0.1))

            transforms_sat.append(transforms.RandomGrayscale(p=0.1))
            transforms_street.append(transforms.RandomGrayscale(p=0.1))

        elif sematic_aug == 'none':
            pass
        else:
            raise RuntimeError(f"sematic augmentation {sematic_aug} is not implemented")

        transforms_sat.append(transforms.ToTensor())
        transforms_sat.append(transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)))

        transforms_street.append(transforms.ToTensor())
        transforms_street.append(transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)))

        self.transforms_sat = transforms.Compose(transforms_sat)
        self.transforms_street = transforms.Compose(transforms_street)

        self.geometric_aug = geometric_aug
        self.sematic_aug = sematic_aug

        if mode == "val" or mode == "dev":
            self.file = os.path.join(self.data_dir, "splits", "val-19zl.csv")
        elif mode == "train":
            self.file = os.path.join(self.data_dir, "splits", "train-19zl.csv")
        else:
            raise RuntimeError("no such mode")

        self.data_list = []
        
        csv_file = open(self.file)
        for l in csv_file.readlines():
            data = l.strip().split(",")
            data.pop(2)
            if is_polar:
                data[0] = data[0].replace("bingmap", "sat")
                # data[0] = data[0].replace("jpg", "png")
            data[1] = data[1].replace('streetview/', '')
            self.data_list.append(data)

        csv_file.close()

        if mode == "dev":
            self.data_list = self.data_list[0:200]

        self.yaw_data = pd.read_csv(f'{data_dir}/split/all.csv', header=None)
        self.yaw_data = self.yaw_data.rename(columns={0: "idx", 4: "yaw"})
        self.yaw_data = self.yaw_data['yaw'].to_list()
            
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        satellite_file, ground_file = self.data_list[index]

        satellite = Image.open(os.path.join(self.data_dir, satellite_file)).convert('RGB')
        ground = Image.open(os.path.join(self.data_dir, f'streetview/{ground_file}')).convert('RGB')

        # Yaw roll and FOV cropping
        query_img = np.array(ground)
        yaw = self.yaw_data[index-1]
        width = query_img.shape[1]
        rollable = (yaw/360) * width * -1
        query_img = np.roll(query_img, int(rollable), axis=1)
        h, w, c = query_img.shape
        half_width = ((self.fov / 360) / 2) * w
        centre_pixel = w // 2
        query_img = query_img[:, int(centre_pixel-half_width):int(centre_pixel+half_width), :]
        ground = Image.fromarray(query_img)

        satellite_first = self.transforms_sat(satellite)
        ground_first = self.transforms_street(ground)

        return satellite_first, ground_first

















            
###### SAT ALIGNMENT #####

def orientation_tensor(height):
    road_width = 6
    alignment_tensor = np.zeros((1, height, height))
    alignment_tensor[:, :int(height/2), int(height/2)-road_width:int(height/2)+road_width] = 1
    np_road = alignment_tensor.squeeze(0)
    tn_road = torch.from_numpy(np_road).float()
    return tn_road

def rotate_seg(seg, deg):
    seg = seg.unsqueeze(0)
    seg = rotate(img=seg, angle=deg, interpolation=Image.NEAREST, expand=True)
    seg = center_crop(seg, (224, 224))
    return seg

def rotate_img(img, deg):
    img = img.unsqueeze(0)
    img = rotate(img=img, angle=deg, interpolation=Image.BILINEAR, expand=True)
    img = center_crop(img, (224, 224))
    return img

pil = T.ToPILImage()
cwd = Path.cwd()

def imagify(img_seg, name='test'):
    img = pil(img_seg)
    img.save(f'{cwd}/outputs/test/{name}.png')

def absolute_180_error(err_list):
    new_errs = []
    for j in err_list:
        i = abs(j[0])
        if i < 90: 
            new_errs.append([i])
        elif i > 90 and i < 270: 
            new_errs.append([abs(180 - i)])
        else: 
            new_errs.append([abs(360 - i)])
    err_mean = torch.mean(torch.tensor(new_errs).float())
    return err_mean


class BEVCV(pl.LightningModule):
    def __init__(self, config=None):
        super(BEVCV, self).__init__()
        self.save_hyperparameters()
        self.cfg = config
        self.current_val_loss = 10
        self.train_labels = torch.arange(self.cfg.batch_size)
        self.test_labels = torch.arange(self.cfg.batch_size)
        self.zero_train_images, self.zero_val_images, self.zero_test_images = 0, 0, 0
        # self.train_path = f'{self.cfg.cv_data}/CVUSA_train_{self.cfg.fov}.beton'#{self.cfg.dataset}_train_{self.cfg.fov}_{self.cfg.resized}.beton'
        # self.val_path = f'{self.cfg.cv_data}/CVUSA_val_{self.cfg.fov}.beton'#{self.cfg.dataset}_val_{self.cfg.fov}_{self.cfg.resized}.beton'
        self.calib_demo = torch.Tensor([[630, 0.0, 694.4383], [0.0, 630, 241.6793], [0.0, 0.0, 1.0]])

        bev_model = BEVNET()
        bev_model.load_state_dict(torch.load(f'/scratch/projects/bev-cv-polish/weights/pov_best.pth')['model'])
        bev_model.freeze()

        self.pov_fpn = bev_model.frontend
        self.pov_transformer = bev_model.transformer
        self.bev_conv = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=3), nn.BatchNorm2d(128), nn.LeakyReLU(),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=3), nn.BatchNorm2d(256), nn.LeakyReLU(),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1), nn.BatchNorm2d(512), nn.LeakyReLU())

        # self.map_unet = smp.UnetPlusPlus(encoder_name="resnet50", in_channels=3, classes=2, activation='sigmoid')
        # self.map_unet.load_state_dict(torch.load(f'{self.cfg.path}/trained_models/map_sem/best_model.pth'))
        # self.map_branch = self.map_unet.encoder

        #  ConvNext encoder
        self.map_branch = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
        self.map_branch.classifier[2] = nn.Identity()

        self.standard_shaper = ShapeStandardiser()
        self.projection = ProjectionModule(config=self.cfg, scale=1)
        self.loss_function = NTXentLoss(temperature=0.10)
        self.normalise = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
 
        self.to_tensor = T.ToTensor()
        self.to_pil = T.ToPILImage()

        self.train_outputs, self.val_outputs, self.test_outputs = [], [], []

        self.sat_db_train, self.sat_db_val, self.sat_db_test = None, None, None
        self.train_corr, self.val_corr, self.test_corr = False, False, False
        self.train_orientations, self.val_orientations, self.test_orientations = None, None, None

        self.train_yaw_err, self.val_yaw_err, self.test_yaw_err = [], [], []

    def forward(self, pov_image, map_tile, calib):
        pov_image, map_tile = self.normalise(pov_image), self.normalise(map_tile)
        # BEV
        feature_maps = self.pov_fpn(pov_image)
        bev_feats = self.pov_transformer(feature_maps, calib)
        pov_embedding = self.bev_conv(bev_feats)
        # Satellite
        map_embedding = self.map_branch(map_tile)

        print(f'Shapes: {pov_embedding.shape}, {map_embedding.shape}')
        breakpoint()

        pov_embedding, map_embedding = self.standard_shaper(pov_embedding, map_embedding)
        # Projection
        pov_embedding, map_embedding = self.projection(pov_embedding, map_embedding)
        return pov_embedding, map_embedding

    def dataloader_function(self, stage='train'):
        dataset = USADataset(mode=stage)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.workers)
        return loader

    def train_dataloader(self): 
        return self.dataloader_function(stage='train')

    def step_function(self, batch, index, step_type='train'):
        pov, sat = batch

        calib = self.calib_demo.repeat(pov.shape[0], 1, 1)
        pov, sat, calib = pov.to(device), sat.to(device), calib.to(device)

        pov_emb, map_emb = self.forward(pov, sat, calib)

        self.labels = torch.arange(pov_emb.shape[0])
        loss = self.loss_function(pov_emb, self.labels, ref_emb=map_emb, ref_labels=self.labels)

        self.log(f'{step_type}_ls', loss.item(), logger=True, batch_size=self.cfg.batch_size)

        if step_type != 'train': #((self.current_epoch + 1) % self.cfg.train_acc_every == 0) or  
            dic = dict(pov=pov_emb.detach().cpu(), map=map_emb.detach().cpu(), loss=loss.item())
            self.val_outputs.append(dic) if step_type == 'val' else self.test_outputs.append(dic)
        if step_type == 'train': 
            self.train_outputs.append(dict(loss=loss.item()))
        return {'loss': loss}

    def training_step(self, batch, batch_idx): return self.step_function(batch, batch_idx, step_type='train')

    def validation_step(self, batch, batch_idx): return self.step_function(batch, batch_idx, step_type='val')

    def test_step(self, batch, batch_idx): return self.step_function(batch, batch_idx, step_type='test')

    def epoch_end_function(self, stage='train', length=35531):
        outputs = getattr(self, f'{stage}_outputs')

        loss = torch.mean(torch.Tensor([item['loss'] for item in outputs]))
        if stage == 'val': 
            self.current_val_loss = loss
        self.log(f'{stage}_epoch_loss', loss, prog_bar=True, logger=True)
        
        if stage != 'train': #((self.current_epoch + 1) % self.cfg.train_acc_every == 0) or 
            pov_embs = torch.cat([item['pov'] for item in outputs], dim=0)
            map_embs = torch.cat([item['map'] for item in outputs], dim=0)
            # if self.cfg.drop_zeros: length = length - zero_images
            recall_rates = accuracy(pov_embs, map_embs, workers=self.cfg.workers, length=pov_embs.shape[0], config=self.cfg)
            for i, acc in enumerate(recall_rates): 
                self.log(f'{stage}_top_{self.cfg.top_ks[i]}', acc, prog_bar=True, logger=True)

        exec(f'self.{stage}_outputs.clear()')
        if stage == 'train': self.zero_train_images = 0
        elif stage == 'val': self.zero_val_images = 0
        else: self.zero_test_images = 0
        # setattr(self, f'{stage}_outputs', [])

    def on_train_epoch_end(self): self.epoch_end_function(stage='train', length=self.train_length)

    def on_validation_epoch_end(self): self.epoch_end_function(stage='val', length=self.val_length)

    def on_test_epoch_end(self): self.epoch_end_function(stage='test', length=self.test_length)

    def configure_optimizers(self): 
        opt = Adam(self.parameters(), lr=self.cfg.learning_rate)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5, verbose=True)
        return {'optimizer': opt, 'lr_scheduler': sch, 'monitor': 'val_epoch_loss'}


if __name__ == '__main__':

    cfg = get_cfg_defaults()
    model = BEVCV(cfg)
    trainer = pl.Trainer(max_epochs=10, devices=1, accelerator='gpu')
    trainer.fit(model)
    
