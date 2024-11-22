import torchvision.transforms as T
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
# from src.data.lmdb_ffcv import Pov_Map_Sem_Dataset, Database
from torch.optim import Adam
import math
import pytorch_lightning as pl
import torch.onnx
import torch
import segmentation_models_pytorch as smp
# from segmentation_models_pytorch.encoders import get_preprocessing_fn
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss, FocalLoss, LovaszLoss, SoftCrossEntropyLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# matplotlib.use('TkAgg')

class MapBranch(pl.LightningModule):
    def __init__(self, config=None):
        super(MapBranch, self).__init__()
        print(f'Config: {config}')
        self.path = config.path
        self.bev = config.bev
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.learning_rate = 0.001
        self.workers = config.workers
        self.data_type = torch.float32
        self.num_epochs = config.epochs
        self.loss_type = config.loss
        self.colour_mappings = {
            'background': (0, 0, 255),
            'buildings': (255, 0, 0),
            'road': (0, 255, 0),
        }
        self.step_number = 0
        self.prepare_data()

        self.model = smp.UnetPlusPlus(
            encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=3,  # model output channels (number of classes in your dataset)
            activation='sigmoid'
        )

        # self.transform_norm = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transform_norm = smp.get_preprocessing_fn('resnet34', pretrained='imagenet')

        if self.loss_type == 'dice':
            self.loss_function = DiceLoss(mode='multilabel')
        elif self.loss_type == 'jaccard':
            self.loss_function = JaccardLoss(mode='multilabel')
        elif self.loss_type == 'focal':
            self.loss_function = FocalLoss(mode='multilabel')
        elif self.loss_type == 'lovasz':
            self.loss_function = LovaszLoss(mode='multilabel')
        elif self.loss_type == 'cross':
            self.loss_function = SoftCrossEntropyLoss(smooth_factor=0)

    def forward(self, map):
        map = self.transform_norm(map)
        out = self.model(map)
        return out

    def prepare_data(self):
        self.images_db_train = Database(path=f'/vol/research/deep_localisation/CV_data/massachusetts/train/images/',
                                        readahead=True)
        self.masks_db_train = Database(path=f'/vol/research/deep_localisation/CV_data/massachusetts/train/masks/',
                                        readahead=True)
        self.images_db_test = Database(path=f'/vol/research/deep_localisation/CV_data/massachusetts/test/images/',
                                        readahead=True)
        self.masks_db_test = Database(path=f'/vol/research/deep_localisation/CV_data/massachusetts/test/masks/',
                                        readahead=True)
        self.keys = list(set(self.images_db_train.keys).intersection(self.masks_db_train.keys))
        self.length = len(self.keys)
        self.test_keys = list(set(self.images_db_test.keys).intersection(self.masks_db_test.keys))
        self.test_length = len(self.test_keys)
        print(f'Dataset Train Length: {self.length}')

    def setup(self, stage=None):
        train_split = math.floor(self.length * 0.9)
        self.train_keys = self.keys[:train_split]
        self.val_keys = self.keys[train_split:]

    def train_dataloader(self):
        if self.dataset == 'custom':
            ds = Pov_Map_Sem_Dataset(pdb=None, mdb=self.map_db_train, batch_size=self.batch_size, semantic=True,
                                     bev=self.bev, subset_keys=self.train_keys)
        elif self.dataset == 'massachusetts':
            ds = Pov_Map_Sem_Dataset(mdb=self.images_db_train, mask_db=self.masks_db_train, batch_size=self.batch_size,
                                     subset_keys=self.train_keys)
        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.workers, pin_memory=True, shuffle=True,
                          drop_last=True)

    def val_dataloader(self):
        if self.dataset == 'custom':
            ds = Pov_Map_Sem_Dataset(pdb=None, mdb=self.map_db_train, batch_size=self.batch_size, semantic=True,
                                     bev=self.bev, subset_keys=self.val_keys)
        elif self.dataset == 'massachusetts':
            ds = Pov_Map_Sem_Dataset(mdb=self.images_db_train, mask_db=self.masks_db_train, batch_size=self.batch_size,
                                     subset_keys=self.val_keys)
        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.workers, pin_memory=True, shuffle=False,
                          drop_last=True)

    def test_dataloader(self):
        if self.dataset == 'custom':
            ds = Pov_Map_Sem_Dataset(pdb=None, mdb=self.map_db_test, batch_size=self.batch_size, semantic=True,
                                     bev=self.bev, subset_keys=self.map_db_test.keys)
        elif self.dataset == 'massachusetts':
            ds = Pov_Map_Sem_Dataset(mdb=self.images_db_test, mask_db=self.masks_db_test, batch_size=self.batch_size,
                                     subset_keys=self.test_keys)
        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.workers, pin_memory=True, shuffle=False,
                          drop_last=True)

    def step_function(self, batch, step_type='train'):
        self.step_number += 1
        # Semantic ground truth is now greyscale with 0-3 classes
        map, sem = batch
        segmentation_pred = self.forward(map)
        loss = self.loss_function(segmentation_pred, sem)
        self.log(f'{step_type}_ls', loss, logger=True, batch_size=self.batch_size)
        # if ((self.current_epoch + 1) % 10) == 0 and self.step_number > math.floor(
        #         (self.length * 0.8) / self.batch_size):
        #     return {'pred': segmentation_pred[0].cpu(), 'gt': sem[0].cpu(), 'loss': loss}
        # else:
        return {'loss': loss}

    def training_step(self, batch, batch_idx):
        return self.step_function(batch, step_type='train')

    # def training_epoch_end(self, outputs):
    #     if ((self.current_epoch + 1) % 10) == 0:  # (int(self.num_epochs / 5))) == 0:
    #         prediction = outputs[-1]['pred']
    #         # ground_truth = outputs[-1]['gt'][0]
    #         pred_img = self.view_semantics(prediction)

    def validation_step(self, batch, batch_idx):
        return self.step_function(batch, step_type='val')

    def validation_epoch_end(self, outputs):
        self.current_val_loss = torch.stack([x['loss'] for x in outputs]).mean()

    def test_step(self, batch, batch_idx):
        return self.step_function(batch, step_type='test')

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return self.forward(*batch)

    def configure_optimizers(self):
        self.opt = Adam(self.parameters(), lr=self.learning_rate)
        self.reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode='min',
            factor=0.9,
            patience=9,
            verbose=True,
            cooldown=0,
            min_lr=1e-9,
        )
        return self.opt

    def optimizer_step(self, epoch=None, batch_idx=None, optimizer=None, optimizer_idx=None, optimizer_closure=None,
                       on_tpu=None, using_native_amp=None, using_lbfgs=None):
        if batch_idx == 0:
            self.reduce_lr_on_plateau.step(self.current_val_loss)

        self.opt.step(closure=optimizer_closure)
        self.opt.zero_grad()

    def view_semantics(self, tensor):
        seg = torch.argmax(tensor, 0).cpu().detach().numpy()

        fig = plt.imshow(seg)
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        fig.figure.savefig(
            f'{self.dir}outputs/wed_pred_BEV{self.bev}_{self.loss_type}_epoch{self.current_epoch}.png',
            bbox_inches='tight',
            pad_inches=0)
        fig.figure.clear()
        plt.close()
        plt.cla()
        plt.clf()
