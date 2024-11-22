from __future__ import print_function
from pathlib import Path
import datetime
import argparse
import torch
import lightning.pytorch as pl
from lightning.pytorch import loggers as plg

from config.config import get_cfg_defaults
from src.models.bevcv import BEVCV

## CONFIGURATION
parser = argparse.ArgumentParser(description='BEV-CV CVGL Network')
parser.add_argument('--config', type=str, default='default/bevcv_cvusa', metavar='N', help='experiment config file')
parser.add_argument('--resume-training', type=bool, default=False, metavar='N', help='Condor flag')
parser.add_argument('--workers', type=int, default=4, metavar='N', help='workers')
parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size')
parser.add_argument('--fov', type=int, default=70, metavar='N', help='batch size')
config = parser.parse_args()
args = vars(config)
dictlist = []
for key, value in list(args.items()): dictlist.extend((key, value))

cfg = get_cfg_defaults()
cfg.merge_from_file(f'{cfg.path}/config/{config.config}.yaml')
cfg.merge_from_list(dictlist)
cfg.freeze()

## SETUP
checkpoint_path = f'{cfg.path}/trained_models/checkpoints/{cfg.log_name}/'
ckpt_dir = Path(checkpoint_path)
ckpt_dir.mkdir(exist_ok=True, parents=True) 
t = datetime.datetime.now()
time_string = "_".join(str(t.time()).split(":")[:3])
pl.seed_everything(cfg.seed)

wandb_logger = plg.WandbLogger(project="BEV-CV", save_dir=f'{cfg.path}/logs/', name=f'{cfg.log_name}',
                               version=f'{cfg.log_name}_{t.date()}_{time_string}', log_model=False)

checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_epoch_loss", mode="min", dirpath=checkpoint_path, save_top_k=1,
                                      filename=f'{cfg.log_name}')

model = BEVCV(config=cfg)
trainer = pl.Trainer(max_epochs=cfg.epochs, devices=1, accelerator='gpu',
                    logger=[wandb_logger], 
                    callbacks=[checkpoint_callback], 
                    check_val_every_n_epoch=cfg.train_acc_every, num_sanity_val_steps=0,
                    # overfit_batches=200
                    )

if cfg.resume_training: 
    if cfg.saved_model != '': trainer.fit(model, ckpt_path=f'{cfg.path}/trained_models/checkpoints/{cfg.saved_model}')
    else: trainer.fit(model, ckpt_path=f'{checkpoint_path}/{cfg.log_name}.ckpt')
else: 
    trainer.fit(model)

torch.save(model, f'{cfg.path}/trained_models/bevcv_{cfg.ex}_{cfg.log_name}_{cfg.fov}.pt')


