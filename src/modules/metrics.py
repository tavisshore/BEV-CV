import numpy as np
import torch
import scipy.spatial as ss
import math
from torchvision.transforms.functional import rotate, center_crop
from PIL import Image


def bev_mask(image):
    left_ones = np.flip(np.tri(M=image.shape[1], N=image.shape[0], k=0, dtype=np.int32))
    right_ones = np.flip(left_ones, axis=1)
    # left_ones = np.stack((left_ones, left_ones, left_ones), axis=2)
    # right_ones = np.stack((right_ones, right_ones, right_ones), axis=2)
    image = image * left_ones
    image = image * right_ones
    image_size = (image.shape[1], image.shape[0])
    left = math.floor(image_size[0] * 0.25)
    right = math.floor(image_size[0] * 0.75)
    bottom = math.floor(image_size[1] * 0.5)
    top = 0
    return image[top:bottom, left:right]

def orientation_tensor(height, bev=True):
    alignment_tensor = np.zeros((1, 224, 224))
    alignment_tensor[:, :height//2, 112-5:112+5] = 1
    np_road = alignment_tensor.squeeze(0)
    if bev: np_road = bev_mask(np_road)
    tn_road = torch.from_numpy(np_road).float()
    return tn_road

def rotate_seg(seg, deg):
    seg = seg.unsqueeze(0)
    seg = rotate(img=seg, angle=deg, interpolation=Image.NEAREST, expand=True)
    seg = center_crop(seg, (112, 112))
    return seg


        
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


