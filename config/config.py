from yacs.config import CfgNode as CN
from pathlib import Path


def get_cfg_defaults():
    # Default configs for BEV-CV 
    _C = CN()
    _C.ex = 0
    _C.epochs = 150
    _C.batch_size = 32
    _C.config = 'default/bevcv_cvusa'
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

