import torch
from torch import optim
import torch.nn.functional as F
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from datasets.data_dsprites import load_dsprites
from datasets.data_teapots import load_teapot
from datasets.data_shapes import load_shapes
from datasets.data_2dshapes import load_2dshapes
from datasets.data_cars import load_3dcars

from torch.utils.data import DataLoader


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        if self.params['dataset'] == 'dsprites':
            self.dl_t, self.dl_v = load_dsprites(root=self.params['data_path'], seed=42)
        elif self.params['dataset'] == 'teapot':
            self.dl_t, self.dl_v = load_teapot(root=self.params['data_path'], seed=42)
        elif self.params['dataset'] == '3dshapes':
            self.dl_t, self.dl_v = load_shapes(root=self.params['data_path'], seed=42)
        elif self.params['dataset'] == '2dshapes':
            self.dl_t, self.dl_v = load_2dshapes(root=self.params['data_path'], seed=42)
        elif self.params['dataset'] == '3dcars':
            self.dl_t, self.dl_v = load_3dcars(root=self.params['data_path'], seed=42)
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):

        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['batch_size'] / self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(*results,
                                            M_N=self.params['batch_size'] / self.num_val_imgs,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.sample_images()
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels=test_label)
        vutils.save_image(recons.data,
                          f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/"
                          f"recons_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True,
                          nrow=12)

        # vutils.save_image(test_input.data,
        #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                   f"real_img_{self.logger.name}_{self.current_epoch}.png",
        #                   normalize=True,
        #                   nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels=test_label)
            vutils.save_image(samples.cpu().data,
                              f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/"
                              f"{self.logger.name}_{self.current_epoch}.png",
                              normalize=True,
                              nrow=12)
        except:
            pass
        del test_input, recons  # , samples

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    @data_loader
    def train_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'dsprites':
            dataset = self.dl_t
        elif self.params['dataset'] == 'teapot':
            dataset = self.dl_t
        elif self.params['dataset'] == '3dshapes':
            dataset = self.dl_t
        elif self.params['dataset'] == '2dshapes':
            dataset = self.dl_t
        elif self.params['dataset'] == '3dcars':
            dataset = self.dl_t
        else:
            raise ValueError('Undefined dataset type')

        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size=self.params['batch_size'],
                          shuffle=True,
                          drop_last=True, num_workers=3)

    @data_loader
    def val_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'dsprites':
            self.sample_dataloader = DataLoader(self.dl_v, batch_size=144,
                                                shuffle=True,
                                                drop_last=True, num_workers=3)
            self.num_val_imgs = len(self.sample_dataloader)

        elif self.params['dataset'] == 'teapot':
            self.sample_dataloader = DataLoader(self.dl_v, batch_size=144,
                                                shuffle=True,
                                                drop_last=True, num_workers=3)
            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] == '3dshapes':
            self.sample_dataloader = DataLoader(self.dl_v, batch_size=144,
                                                shuffle=True,
                                                drop_last=True, num_workers=3)
            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] == '2dshapes':
            self.sample_dataloader = DataLoader(self.dl_v, batch_size=144,
                                                shuffle=True,
                                                drop_last=True, num_workers=3)
            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] == '3dcars':
            self.sample_dataloader = DataLoader(self.dl_v, batch_size=144,
                                                shuffle=True,
                                                drop_last=True, num_workers=3)
            self.num_val_imgs = len(self.sample_dataloader)
        else:
            raise ValueError('Undefined dataset type')

        return self.sample_dataloader

    def data_transforms(self):

        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X / X.sum(0).expand_as(X))

        if self.params['dataset'] == 'celeba':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])
        elif self.params['dataset'] == 'dsprites':
            transform = transforms.Compose([transforms.ToTensor(),
                                            SetRange])
        elif self.params['dataset'] == 'teapot':
            transform = transforms.Compose([transforms.ToTensor(),
                                            SetRange])
        elif self.params['dataset'] == '3dshapes':
            transform = transforms.Compose([transforms.ToTensor(),
                                            SetRange])
        elif self.params['dataset'] == '2dshapes':
            transform = transforms.Compose([transforms.ToTensor(),
                                            SetRange])
        elif self.params['dataset'] == '3dcars':
            transform = transforms.Compose([transforms.ToTensor(),
                                            SetRange])
        else:
            raise ValueError('Undefined dataset type')
        return transform
