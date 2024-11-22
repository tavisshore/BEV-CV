from PIL import Image
import math
import cv2
import numpy as np
import torchvision.transforms as T
import torch
import pytorch_lightning as pl
from pathlib import Path

# from ffcv.fields.decoders import NDArrayDecoder
# from ffcv.transforms import ToTensor


def rotate_image(image, angle):
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)
    rot_mat = np.vstack([cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])
    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5
    rotated_coords = [(np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0], (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0], 
                      (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0], (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]]
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]
    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]
    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)
    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))
    trans_mat = np.matrix([[1, 0, int(new_w * 0.5 - image_w2)], [0, 1, int(new_h * 0.5 - image_h2)], [0, 0, 1]])
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]
    result = cv2.warpAffine(image, affine_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)
    return result


def crop_center(img, cropx, cropy):
    _, y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[:, starty:starty + cropy, startx:startx + cropx]

def largest_rotated_rect(w, h, angle):
    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi
    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)
    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)
    delta = math.pi - alpha - gamma
    length = h if (w < h) else w
    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)
    y = a * math.cos(gamma)
    x = y * math.tan(gamma)
    return (bb_w - 2 * x, bb_h - 2 * y)

def crop_around_center(image, width, height):
    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))
    if width > image_size[0]:
        width = image_size[0]
    if height > image_size[1]:
        height = image_size[1]
    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)
    return image[y1:y2, x1:x2]

def add_padding(pil_img, top, bottom):
    width, height = pil_img.size
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (width, new_height), (0, 0, 0))
    result.paste(pil_img, (0, top))
    return result

resize = T.Resize((224, 224), antialias=True)

def tensor_to_numpy(input_tensor):
    if type(input_tensor) == torch.Tensor:
        if input_tensor.is_cuda: input_tensor = input_tensor.cpu()
        input_tensor = input_tensor.numpy()
    return input_tensor


def map_rotate_database(map_img, angle):
    image = tensor_to_numpy(map_img)
    map_img = rotate_image(image, angle)
    map_img = torch.tensor(crop_around_center(map_img, 512, 512))
    return map_img


def gal_database(satellite_batch, rotations):
    # List of angles for rotating images
    angles = [int((360/rotations)*i) for i in range(0, rotations)]
    rots = []
    for satellite_image in satellite_batch:
        for angle in angles:
            rotated_image = map_rotate_database(satellite_image, angle)
            if rotated_image.shape[0] != 3: rotated_image = rotated_image.permute(2, 0, 1)
            resize_image = resize(rotated_image)
            rots.append(resize_image)
    stacked = torch.stack(rots).float()
    return stacked

#### MOST RECENT INPUT Cropped POV, north-aligned map, heading

class FOVCrop(pl.LightningModule):
    def __init__(self, **kwargs):
        super(FOVCrop, self).__init__()
        for key, value in kwargs.items(): setattr(self, key, value)
        self.resize = T.Resize((224, 224), antialias=True)
        self.resize_pov = T.Resize((512, 128), antialias=True)
        self.to_pil = T.ToPILImage()
        self.to_ten = T.ToTensor()

    # def pov_crop(self, pov, head):
    #     pano_height, pano_width, _ = pov.shape
    #     if pano_width < 512:
    #         pov = self.to_ten(self.to_pil(pov))
    #         return self.resize(pov)
    #     else:
    #         crop_half_width = math.floor(((self.fov / 360) * pano_width) / 2)
        
    #         half = math.floor(pano_width / 2)
    #         heading_pixel = math.floor((head / 360) * pano_width)

    #         if heading_pixel > crop_half_width and heading_pixel < (pano_width - crop_half_width):
    #             centre_point = heading_pixel
    #             image = self.to_pil(pov)
    #         else:
    #             left_img = (0, 0, half, pano_height)
    #             right_img = (half, 0, pano_width, pano_height)
    #             img = self.to_pil(pov)
    #             left_cropped = np.array(img.crop(left_img))
    #             right_cropped = np.array(img.crop(right_img))
    #             image = np.concatenate((right_cropped, left_cropped), axis=1)
    #             image = self.to_pil(image)
    #             if head < self.fov: centre_point = ((head + 180) / 360) * pano_width
    #             else: centre_point = ((head - 180) / 360) * pano_width
            
    #         left = math.floor(centre_point - crop_half_width)
    #         right = math.floor(centre_point + crop_half_width)
    #         area = (left, 0, right, pano_height)
    #         return self.resize(self.to_ten(image.crop(area)))

    def map_crop(self, map_img, head):
        if head != 0: map_img = map_rotate_database(map_img, head)
        else: map_img = torch.tensor(crop_around_center(map_img, 512, 512))
        resized = self.resize(map_img.permute(2, 0, 1))
        return resized

    def forward(self, map_in, heading, stage): # pov_in
        maps = []
        batch_size = map_in.shape[0]

        for image in range(batch_size): # method without for loop?
            yaw = int(heading[image])

            if stage == 'train':
                if self.aug > 0: yaw += torch.randn(1, device=self.device) * self.aug # int?

            if type(map_in) == torch.Tensor: map_in = map_in.detach().cpu().numpy()   
            # povs.append(self.pov_crop(pov_in[image], int(pov_head)))
            maps.append(self.map_crop(map_in[image], int(yaw)))

        # povs = torch.stack(povs).float()
        maps = torch.stack(maps).float()
        return maps # povs, maps



#### Config utils

def data_params(config):
    # path
    if config.system == 'stornext': path = Path(config.stornext.cvusa) if config.dataset == 'cvusa' else Path(config.stornext.cvact)
    elif config.system == 'weka': path = Path(config.weka.cvusa) if config.dataset == 'cvusa' else Path(config.weka.cvact)
    else: path = Path(config.local.cvusa) if config.dataset == 'cvusa' else Path(config.local.cvact)

    # pipeline
    # if config.dataset == 'cvusa': 
        # pipeline = {'pano': [NDArrayDecoder(), ToTensor()],
                    # 'sat': [NDArrayDecoder(), ToTensor()]}
    return path
# , pipeline




