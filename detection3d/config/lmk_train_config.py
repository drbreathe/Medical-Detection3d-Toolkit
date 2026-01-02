from easydict import EasyDict as edict
from detection3d.utils.normalizer import FixedNormalizer, AdaptiveNormalizer

__C = edict()
cfg = __C

##################################
# general parameters
##################################
__C.general = {}

__C.general.training_image_list_file = '/home/mohanty/VSCODE_PROJECTS/Medical-Detection3d-Toolkit-bk/detection3d/data/dataset_lmk_33/train_fold1.csv'

__C.general.validation_image_list_file = '/home/mohanty/VSCODE_PROJECTS/Medical-Detection3d-Toolkit-bk/detection3d/data/dataset_lmk_33/val_fold1.csv'

# landmark label starts from 1, 0 represents the background.
__C.general.target_landmark_label = {
    "Or-Left": 1,
    "Or-Right": 2,
    "Po-Left": 3,
    "Po-Right": 4,
    "Nasion": 5,
    "S": 6,
    "Ba": 7,
    "ANS": 8,
    "A": 9,
    "PNS": 10,
    "CI": 11,
    "J-Left": 12,
    "J-Right": 13,
    "J_PVA": 14,
    "B": 15,
    "Pog": 16,
    "Me": 17,
    "Gn": 18,
    "Go_Inf-Left": 19,
    "Go_Inf-Right": 20,
    "Go_Sup-Left": 21,
    "Go_Sup-Right": 22,
    "Co_Pos-Left": 23,
    "Co_Pos-Right": 24,
    "Co_Sup-Left": 25,
    "Co_Sup-Right": 26,
    "H": 27,
    "Sn": 28,
    "Cm": 29,
    "Ls": 30,
}

__C.general.save_dir = './saves/weights'

__C.general.resume_epoch = -1

__C.general.num_gpus = 1

##################################
# dataset parameters
##################################
__C.dataset = {}

__C.dataset.crop_spacing = [0.5, 0.5, 0.5]      # mm

__C.dataset.crop_size = [128, 128, 128]   # voxel

__C.dataset.pad_size = [8, 8, 8]   # voxel, must be multiple of stride

__C.dataset.sampling_size = [48, 48, 48]      # voxel

__C.dataset.positive_upper_bound = 3    # voxel

__C.dataset.negative_lower_bound = 6

__C.dataset.num_pos_patches_per_image =  32 # This should not be same as Number of landmarks...Rather i should be less than number of positive pertubes

__C.dataset.num_neg_patches_per_image = 24

# crop intensity normalizers (to [-1,1])
# one normalizer corresponds to one input modality
# 1) FixedNormalizer: use fixed mean and standard deviation to normalize intensity
# 2) AdaptiveNormalizer: use minimum and maximum intensity of crop to normalize intensity
__C.dataset.crop_normalizers = [AdaptiveNormalizer()]

# sampling method:
# 1) GLOBAL : sampling crops randomly in the entire image domain
# 2) LOCAL_TO_LMK : sampling crops near the vicinity of the landmark
# 3) HYBRID : sampling crops Globally 30% of the time and Local to landmark 70% of the time
__C.dataset.sampling_method = 'HYBRID'

# linear interpolation method:
# 1) NN: nearest neighbor interpolation
# 2) LINEAR: linear interpolation
__C.dataset.interpolation = 'LINEAR'

##################################
# data augmentation parameters
##################################

__C.augmentation = {}

__C.augmentation.turn_on = True

__C.augmentation.translation_lmk = 5
# ------------------ Affine ------------------
__C.augmentation.affine_turn_on = True

__C.augmentation.scales = [0.9, 1.1]    # isotropic scale range

__C.augmentation.rotation = 10   # NOTE: despite the name, this is in degrees!

__C.augmentation.translation = 5  # mm

__C.augmentation.affine_p = 0.5 

# ------------------ Flip ------------------
__C.augmentation.flip_turn_on = True

__C.augmentation.flip_p = 0.5

# ------------------ Elastic deformation ------------------
__C.augmentation.elastic_turn_on = True

__C.augmentation.elastic_num_control_points = 4

__C.augmentation.elastic_max_displacement = 3.0   # mm

__C.augmentation.elastic_locked_borders = 1

__C.augmentation.elastic_p = 0.5

# ------------------ Motion ------------------
__C.augmentation.motion_turn_on = True

__C.augmentation.motion_num_transforms = 1

__C.augmentation.motion_p = 0.5

# ------------------ Noise ------------------
__C.augmentation.noise_turn_on = True

__C.augmentation.noise_mean = 0.0

__C.augmentation.noise_std = 0.04

__C.augmentation.noise_p = 0.5

# ------------------ Gamma ------------------
__C.augmentation.gamma_turn_on = True

__C.augmentation.log_gamma = [-0.2, 0.2]

__C.augmentation.gamma_p = 0.4

##################################
# loss function
##################################
__C.landmark_loss = {}

__C.landmark_loss.name = 'Focal'          # 'Dice', or 'Focal'

__C.landmark_loss.focal_obj_alpha = [0.75] * (len(__C.general.target_landmark_label)+1)

__C.landmark_loss.focal_gamma = 2         # gamma in pow(1-p,gamma) for focal loss

##################################
# net
##################################
__C.net = {}

__C.net.name = 'vdnet'

##################################
# training parameters
##################################
__C.train = {}

__C.train.use_amp = True

__C.train.epochs = 3001

__C.train.batch_size = 4

__C.train.num_threads = 8

__C.train.lr = 1e-4

__C.train.weight_decay = 1e-4

__C.train.betas = (0.9, 0.999)

__C.train.save_epochs = 5

__C.train.grad_accum_steps = 2

__C.train.early_stop_patience = 100      # epochs to wait without improvement
__C.train.early_stop_min_delta = 1e-7   # minimum val loss improvement
__C.train.early_stop_start_epoch = 50   # do not early-stop too early


##################################
# validation parameters
##################################
__C.val = {}

__C.val.interval = 1

__C.val.batch_size = 4

__C.val.num_threads = 4

__C.val.eval_fraction = 1

##################################
# debug parameters
##################################
__C.debug = {}

# random seed used in training
__C.debug.seed = 0

# whether to save input crops
__C.debug.save_inputs = False
