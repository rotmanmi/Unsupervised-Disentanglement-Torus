from .base import BaseVAE
from .vanilla_vae import VanillaVAE
from .gamma_vae import GammaVAE
from .beta_vae import BetaVAE
from .wae_mmd import WAE_MMD
from .cvae import ConditionalVAE
from .hvae import HVAE
from .vampvae import VampVAE
from .iwae import IWAE
from .dfcvae import DFCVAE
from .mssim_vae import MSSIMVAE
from .fvae import FactorVAE
from .cat_vae import CategoricalVAE
from .joint_vae import JointVAE
from .info_vae import InfoVAE
# from .twostage_vae import *
from .lvae import LVAE
from .logcosh_vae import LogCoshVAE
from .swae import SWAE
from .miwae import MIWAE
from .vq_vae import VQVAE
from .betatc_vae import BetaTCVAE
from .dip_vae import DIPVAE
from .torus_vae import TorusVAE

# Aliases
VAE = VanillaVAE
GaussianVAE = VanillaVAE
CVAE = ConditionalVAE
GumbelVAE = CategoricalVAE

vae_models = {'HVAE': HVAE,
              'LVAE': LVAE,
              'IWAE': IWAE,
              'SWAE': SWAE,
              'MIWAE': MIWAE,
              'VQVAE': VQVAE,
              'DFCVAE': DFCVAE,
              'DIPVAE': DIPVAE,
              'BetaVAE': BetaVAE,
              'InfoVAE': InfoVAE,
              'WAE_MMD': WAE_MMD,
              'VampVAE': VampVAE,
              'GammaVAE': GammaVAE,
              'MSSIMVAE': MSSIMVAE,
              'JointVAE': JointVAE,
              'BetaTCVAE': BetaTCVAE,
              'FactorVAE': FactorVAE,
              'LogCoshVAE': LogCoshVAE,
              'VanillaVAE': VanillaVAE,
              'ConditionalVAE': ConditionalVAE,
              'CategoricalVAE': CategoricalVAE,
              'TorusVAE': TorusVAE,}
