import yaml
import argparse
import torch
import numpy as np
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/vae.yaml')
parser.add_argument('--latentdim', '-d',
                    help='latent dimension',
                    default=10, type=int)
parser.add_argument('--one', '-o',
                    help='one std and mu per circle',
                    action='store_true')

parser.add_argument('--ns', '-ns',
                    help='Number of S_1 for representation',
                    default=3, type=int)

parser.add_argument('--beta', '-b',
                    help='Value of gamma/beta',
                    default=1.0, type=float)

parser.add_argument('--resnet', '-r',
                    help='Use resnet',
                    action='store_true')

parser.add_argument('--seed', '-s',
                    help='seed',
                    default=1265, type=int)

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

config['logging_params']['name'] = '{}{}'.format(config['logging_params']['name'], args.latentdim)
config['model_params']['latent_dim'] = args.latentdim
config['model_params']['resnet'] = args.resnet
config['logging_params']['manual_seed'] = args.seed
if config['model_params']['resnet']:
    config['logging_params']['name'] += '_ResNet'
if 'bond_dim' in config['model_params'].keys():
    config['model_params']['bond_dim'] = args.latentdim // 8
    config['logging_params']['name'] = '{}_{}'.format(config['logging_params']['name'], args.ns)
    config['logging_params']['name'] += 'one' if args.one else ''
    config['model_params']['one'] = args.one
    config['model_params']['gamma'] = args.beta
    if args.beta != 1.0:
        config['logging_params']['name'] = '{}_{}'.format(config['logging_params']['name'], args.beta)

tt_logger = TestTubeLogger(
    save_dir=os.path.join(config['logging_params']['save_dir'], config['exp_params']['dataset']),
    name=config['logging_params']['name'],
    debug=False,
    create_git_tag=False,
    version=config['logging_params']['manual_seed']
)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

config['model_params']['n_s'] = args.ns

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params'])

checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(config['logging_params']['save_dir'], config['exp_params']['dataset'],
                         config['logging_params']['name'],
                         'version_{}'.format(tt_logger.experiment.version), 'checkpoints'),
    # save_best_only=True,
    verbose=True,
    monitor='Reconstruction_Loss',
    mode='min',
    save_last=True
)

runner = Trainer(  # default_save_path=f"{tt_logger.save_dir}",
    logger=tt_logger,
    num_sanity_val_steps=5,
    callbacks=checkpoint_callback,
    terminate_on_nan=True,
    **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment)
