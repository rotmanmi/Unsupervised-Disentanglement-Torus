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
from metrics.sampling import estimate_SEP_cupy, estimate_SEPIN_cupy
from metrics.dci import dci
from torchvision.utils import save_image
from models.utils import calc_latend_dim, tensor_prod, ind_comps, calc_latent_dim_sampling
import torchvision
import argparse
import utils
import os
from glob import glob
import ast

import logging
import colorama
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.utils
from tkinter import *
from PIL import ImageTk, Image
from torchvision.utils import save_image

root = Tk()
root.geometry("640x1200")

clear = colorama.Style.RESET_ALL
blue = colorama.Fore.CYAN + colorama.Style.BRIGHT
green = colorama.Fore.GREEN + colorama.Style.BRIGHT
magenta = colorama.Fore.MAGENTA + colorama.Style.BRIGHT

np.random.seed(1234)

if __name__ == '__main__':

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
    parser.add_argument('--ns', '-ns',
                        help='Number of S_1 for representation',
                        default=3, type=int)

    parser.add_argument('--one', '-o',
                        help='one std and mu per circle',
                        action='store_true')

    parser.add_argument('--resnet', '-r',
                        help='Use resnet',
                        action='store_true')

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
        config['logging_params']['name'] += 'one' if args.one else ''
        config['model_params']['one'] = args.one

    # For reproducibility
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    cudnn.deterministic = True
    cudnn.benchmark = False
    config['model_params']['n_s'] = args.ns

    model = vae_models[config['model_params']['name']](**config['model_params'])
    checkpoint_path = os.path.join(config['logging_params']['save_dir'], config['exp_params']['dataset'],
                                   config['logging_params']['name'],
                                   'version_{}'.format(args.version), 'checkpoints')
    print(checkpoint_path)

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

    f = Frame(root)
    f.grid(row=1, column=2)
    image_no = ImageTk.PhotoImage(torchvision.transforms.ToPILImage()(
        torchvision.utils.make_grid(
            experiment.model.decode(torch.zeros(1, calc_latent_dim_sampling(args.latentdim // 8, args.ns)).cuda(),
                                    sampling=True), 3,
            normalize=True)).resize((256, 256)))
    label_img = Label(image=image_no)
    label_img.grid(row=4, column=0, columnspan=1)


    def callback_scale(e):
        z = torch.tensor([[c.get() for c in channels_vars]]).cuda()
        # z = torch.tensor([[channels_vars[0].get()] + [np.cos(channels_vars[1].get()), np.cos(channels_vars[2].get()),
        #                                               np.cos(channels_vars[3].get()),
        #                                               np.sin(channels_vars[1].get()), np.sin(channels_vars[2].get()),
        #                                               np.sin(channels_vars[3].get())]]).float().cuda()
        generated_image = experiment.model.decode(z, sampling=True)
        new_image = ImageTk.PhotoImage(
            torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(generated_image, 3, normalize=True)).resize(
                (256, 256)))
        label_img.configure(image=new_image)
        label_img.image = new_image


    channels_vars = [DoubleVar() for _ in range(calc_latent_dim_sampling(args.latentdim // 8, args.ns))]
    [cv.set(0) for cv in channels_vars]
    channel_scales = [Scale(root, from_=-100.0, to=100.0, resolution=0.001, digits=4, orient=HORIZONTAL,
                            length=300,
                            variable=channels_vars[0],
                            command=callback_scale, label='Channel {:02d}'.format(0))] + [
                         Scale(root, from_=0, to=2 * np.pi, resolution=0.001, digits=4, orient=HORIZONTAL,
                               length=300,
                               variable=channels_vars[i],
                               command=callback_scale, label='Channel {:02d}'.format(i)) for i in
                         range(1, len(channels_vars))]
    [channel_scales[i].grid(row=i, column=1) for i in range(len(channels_vars))]

    # channel_var = IntVar()
    # channel_var.set(0)
    # channel = Scale(root, from_=0, to=128, tickinterval=10, orient=HORIZONTAL, length=300,
    #                 variable=channel_var,
    #                 command=callback_scale, label='Channel')
    #
    # amplitude_var = IntVar()
    # amplitude_var.set(1)
    # amplitude = Scale(root, from_=0, to=1000, tickinterval=10, orient=HORIZONTAL, length=300,
    #                   variable=amplitude_var,
    #                   command=callback_scale, label='Channel Amplitude')
    #
    # x_var = IntVar()
    # x_var.set(0)
    # x = Scale(root, from_=0, to=20, tickinterval=10, orient=HORIZONTAL, length=300,
    #           variable=x_var,
    #           command=callback_scale, label='X')
    #
    # y_var = IntVar()
    # y_var.set(0)
    # y = Scale(root, from_=0, to=20, tickinterval=10, orient=HORIZONTAL, length=300,
    #           variable=y_var,
    #           command=callback_scale, label='Y')

    # amplitude.grid(row=1, column=0)
    # x.grid(row=2, column=0)
    # y.grid(row=3, column=0)
    # channel.grid(row=0, column=0)

    root.focus_set()
    root.mainloop()
