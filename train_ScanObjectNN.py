

# Common libs
import ctypes
import signal
import os
import numpy as np
import sys
import torch

# Dataset
import torchvision

from datasets.ScanObjectNN import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.trainer import ModelTrainer
from models.architectures import KPCNN
from models.dataset import ProjectFromPointsOnSphere


# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#

class ScanObjectNNConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'ScanObject'

    # Number of classes in the dataset (This value is overwritten by dataset class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = ''

    # Number of CPU threads for the input pipeline
    input_threads = 10

    #########################
    # Architecture definition
    #########################

    # Define layers

    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'global_average']

    ###################
    # KPConv parameters
    ###################

    # Number of kernel points
    num_kernel_points = 15

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.02

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    deform_radius = 6.0

    # Radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)
    KP_extent = 1.2

    # Behavior of convolutions in ('constant', 'linear', 'gaussian')
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    aggregation_mode = 'sum'

    # Choice of input features
    in_features_dim = 1

    # Can the network learn modulations
    modulated = True

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.05

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0  # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.2  # Distance of repulsion for deformed kernel points

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 1000

    # Learning rate management
    learning_rate = 1e-2
    momentum = 0.98
    lr_decays = {i: 0.1 ** (1 / 100) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0

    # Number of batch
    batch_num = 10

    # Number of steps per epochs
    epoch_steps = 300

    # Number of validation examples per epoch
    validation_size = 30

    # Number of epoch between each checkpoint
    checkpoint_gap = 50

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, True, True]
    augment_rotation = 'none'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    augment_color = 1.0

    # The way we balance segmentation loss
    #   > 'none': Each point in the whole batch has the same contribution.
    #   > 'class': Each class has the same contribution (points are weighted according to class balance)
    #   > 'batch': Each cloud in the batch has the same contribution (points are weighted according cloud sizes)
    segloss_balance = 'none'

    # Do we nee to save convergence
    saving = True
    saving_path = None


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    ###############
    # Previous chkp
    ###############

    # Choose here if you want to start training from a previous snapshot (None for new training)
    previous_training_path = ''
    #previous_training_path = 'Log_2020-10-23_14-54-26'

    # Choose index of checkpoint to start from. If None, uses the latest chkp
    chkp_idx = None
    if previous_training_path:

        # Find all snapshot in the chosen training folder
        chkp_path = os.path.join('results', previous_training_path, 'checkpoints')
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

        # Find which snapshot to restore
        if chkp_idx is None:
            chosen_chkp = 'current_chkp.tar'
        else:
            chosen_chkp = np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join('results', previous_training_path, 'checkpoints', chosen_chkp)

    else:
        chosen_chkp = None

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    # Initialize configuration class
    config = ScanObjectNNConfig()
    if previous_training_path:
        config.load(os.path.join('results', previous_training_path))
        config.saving_path = None

    # Get path from argument if given
    if len(sys.argv) > 1:
        config.saving_path = sys.argv[1]

    from models.spherical_model import init_bandwidth as bw

    _file = 'libmatrix.so'
    _path = './models/' + _file
    lib = ctypes.cdll.LoadLibrary(_path)
    """
    transform = torchvision.transforms.Compose(
    [
        # ToMesh(random_rotations=True, random_translation=0.1),#transform data to mesh
        # ToPoints(random_rotations=True, random_translation=0.1),
        # need to be modified. Originally, the value on the sp  here is based on the ray cast from
        # the points on spherical surface, since that we want to process point cloud data directly,
        # we can try to modified the script, let the ray cast from the point cloud to sphere.
        # ProjectOnSphere(bandwidth=bw)
        ProjectFromPointsOnSphere(bandwidth=bw, lib=lib)
    ]
    )
    """
    transform = ProjectFromPointsOnSphere(bandwidth=bw, lib=lib)
    # Initialize datasets
    training_dataset = ScanObjectNNDataset(config, train=True, spherical_transform=transform)
    test_dataset = ScanObjectNNDataset(config, train=False, spherical_transform=transform)

    # Initialize samplers
    training_sampler = ScanObjectNNSampler(training_dataset, balance_labels=True)
    test_sampler = ScanObjectNNSampler(test_dataset, balance_labels=False)

    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_size=1,
                                 sampler=training_sampler,
                                 collate_fn=ScanObjectNNCollate,
                                 num_workers=config.input_threads,
                                 pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=ScanObjectNNCollate,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    training_sampler.calibration(training_loader)
    test_sampler.calibration(test_loader)

    # debug_timing(test_dataset, test_sampler, test_loader)
    # debug_show_clouds(training_dataset, training_sampler, training_loader)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    net = KPCNN(config)
    net = torch.nn.DataParallel(net)
    # Define a trainer class
    trainer = ModelTrainer(net, config, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart training')
    print('**************')

    # Training
    try:
        trainer.train(net, training_loader, test_loader, config)
    except:
        print('Caught an error')
        os.kill(os.getpid(), signal.SIGINT)

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)
