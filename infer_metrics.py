import torch
import torch.nn.functional as F
import yaml
import argparse
import numpy as np
import os
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from torch.utils.data import SubsetRandomSampler, DataLoader
from tqdm import tqdm
from metrics.dci import dci
import ray
import json, atexit

ray.init()

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

if os.path.exists(os.path.join(checkpoint_path, 'cache.npz')) and not args.debug:
    f = np.load(os.path.join(checkpoint_path, 'cache.npz'))
    all_mu = f['all_mu']
    factors = f['factors']
else:
    model = vae_models[config['model_params']['name']](**config['model_params'])
    model_path = list(Path(checkpoint_path).glob('*.ckpt'))[-1]
    experiment = VAEXperiment(model, config['exp_params']).cuda()
    s = torch.load(model_path)
    experiment.load_state_dict(s['state_dict'])
    dataloader = experiment.val_dataloader()
    all_mu = []
    all_logvar = []
    recon_losses = []
    factors = []
    model.eval()

    indices = list(range(len(experiment.dl_v)))
    np.random.seed(1234)
    np.random.shuffle(indices)
    val_dl = DataLoader(experiment.dl_v, batch_size=128)
    with torch.no_grad():
        with tqdm(total=len(val_dl)) as pbar:
            for i, (x, y) in enumerate(val_dl):
                results = experiment(x.cuda())
                if len(results) == 4:
                    out, _, mu, logvar = results
                else:
                    out, _, mutag, logvar, mu = results
                all_mu.append(mu.detach().cpu())
                all_logvar.append(logvar.detach().cpu())
                recon_losses.extend([F.mse_loss(x.cuda(), out.detach()).item()] * len(x))
                factors.append(y.cpu().detach())
                pbar.update()
    del experiment
    with open(os.path.join(checkpoint_path, 'metrics_RECONLOSS.yaml'), 'w') as file:
        yaml.dump({'Recon_loss': float(np.mean(recon_losses))}, file, default_flow_style=False)
    del mu, logvar
    all_mu = torch.cat(all_mu, 0).cpu().numpy()
    all_logvar = torch.sqrt(torch.exp(torch.cat(all_logvar, 0))).cpu().numpy()
    if len(factors) > 0:
        factors = torch.cat(factors, 0).cpu().numpy()

    np.savez(os.path.join(checkpoint_path, 'cache.npz'), all_mu=all_mu, factors=factors)

if len(factors) > 0:
    print('calculating DCI')
    for reg_model in ['lasso']:
        # First parameter in latent space is dummy and therefore ignored.
        if 'TorusVAE' in config['logging_params']['name']:
            disentanglement, completeness, informativeness, e_matrix = dci(factors, all_mu[:, 1:], model=reg_model)
        else:
            disentanglement, completeness, informativeness, e_matrix = dci(factors, all_mu, model=reg_model)
        coords = np.meshgrid(np.arange(np.shape(e_matrix)[0]),
                             np.arange(np.shape(e_matrix)[1]))
        data = np.vstack([coords[0].flatten(), coords[1].flatten(), e_matrix.flatten()]).T
        results = {'disentanglement': disentanglement.item(), 'completeness': completeness.item(),
                   'informativeness': informativeness.item()}
        with open(os.path.join(checkpoint_path, 'metrics_DCI_{}.yaml'.format(reg_model)), 'w') as file:
            yaml.dump(results, file, default_flow_style=False)
        print(results)
