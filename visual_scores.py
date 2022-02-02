import torch
import torch.nn.functional as F
import yaml
import argparse
import numpy as np
import os
from pathlib import Path

from fid_score import calculate_fid_given_tensors
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from torch.utils.data import SubsetRandomSampler, DataLoader
from tqdm import tqdm
from metrics.dci import dci
import json, atexit
import torchvision.utils as vutils

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/vae.yaml')
parser.add_argument('--version', '-v',
                    dest="version",
                    help='version to load',
                    default='0')

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

parser.add_argument('--debug', action='store_true', help='debug mode')

args = parser.parse_args()
print(args.filename)
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
config['logging_params']['name'] = '{}{}'.format(config['logging_params']['name'], args.latentdim)
config['model_params']['latent_dim'] = args.latentdim
config['model_params']['resnet'] = args.resnet
if config['model_params']['resnet']:
    config['logging_params']['name'] += '_ResNet'
if 'bond_dim' in config['model_params'].keys():
    config['model_params']['bond_dim'] = args.latentdim // 8
    config['logging_params']['name'] = '{}_{}'.format(config['logging_params']['name'], args.ns)
    config['model_params']['gamma'] = args.beta
    config['logging_params']['name'] += 'one' if args.one else ''
    config['model_params']['one'] = args.one
    if args.beta != 1.0:
        config['logging_params']['name'] = '{}_{}'.format(config['logging_params']['name'], args.beta)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False
config['model_params']['n_s'] = args.ns
checkpoint_path = os.path.join(config['logging_params']['save_dir'], config['exp_params']['dataset'],
                               config['logging_params']['name'],
                               'version_{}'.format(args.version), 'checkpoints')
print(checkpoint_path)

model = vae_models[config['model_params']['name']](**config['model_params'])

model_path = list(Path(checkpoint_path).glob('*.ckpt'))[-1]
experiment = VAEXperiment(model, config['exp_params']).cuda()
s = torch.load(model_path)

experiment.load_state_dict(s['state_dict'])

val_dl = DataLoader(experiment.dl_v, batch_size=128)
orig_images = []
with tqdm(total=len(val_dl)) as pbar:
    for i, (x, _) in enumerate(val_dl):
        orig_images.append(x)
        pbar.update()
orig_images = torch.cat(orig_images, 0)
with torch.no_grad():
    imgs = torch.cat([experiment.model.sample(32, 'cuda') for i in range(100)], 0)

FID_score_2048 = calculate_fid_given_tensors(orig_images, imgs, device='cuda', batch_size=32, dims=2048).item()
results = {'fid_score_2048': FID_score_2048}
with open(os.path.join(checkpoint_path, 'metrics_visual.yaml'), 'w') as file:
    yaml.dump(results, file, default_flow_style=False)
print(FID_score_2048)
