# Code Supplementary for [Unsupervised Disentanglement with Tensor Product Representations on the Torus](https://openreview.net/forum?id=neqU3HWDgE) (ICLR 2022) 
The code is heavily based on the [PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE) package
1. First change the "data_path" field in the yaml of the configs directory to point to the folder containing the data.

## In order to run an experiment:


for the TorusVAE with {ns} circles method (ours): 
ns = torus topology, for our runs we used 8.
r = use the resnet architecture
DATASET = 3dcars,teapot,2dshapes,3dshapes,dsprites
```bash
python run.py -c configs/evae_${DATASET}_cosine.yaml -d 8 -ns 8 -r --seed 1265
```

for the baseline methods:
for latent space with D=128
```bash
python run.py -c configs/dip_vae_${DATASET}.yaml -d 128 -r --seed 1265
python run.py -c configs/factorvae_${DATASET}.yaml -d 128 -r --seed 1265
python run.py -c configs/bbvae_${DATASET}.yaml -d 128 -r --seed 1265
```
for latent space with D=10
```bash
python run.py -c configs/dip_vae_${DATASET}.yaml -d 10 -r  --seed 1265
python run.py -c configs/factorvae_${DATASET}.yaml -d 10 -r --seed 1265
python run.py -c configs/bbvae_${DATASET}.yaml -d 10 -r --seed 1265
```

In order to obtain the metrics, after training the models execute:
```bash
python infer_metrics.py -c configs/evae_${DATASET}_cosine.yaml -d 8 -ns 8 -r 
python infer_metrics.py -c configs/dip_vae_${DATASET}.yaml -d 10 -r 
python infer_metrics.py -c configs/factorvae_${DATASET}.yaml -d 10 -r
python infer_metrics.py -c configs/bbvae_${DATASET}.yaml -d 10 -r 
python infer_metrics.py -c configs/dip_vae_${DATASET}.yaml -d 128 -r 
python infer_metrics.py -c configs/factorvae_${DATASET}.yaml -d 128 -r
python infer_metrics.py -c configs/bbvae_${DATASET}.yaml -d 128 -r
```

In order to obtain the FID score run
```bash
python visual_scores.py -c configs/evae_${DATASET}_cosine.yaml -d 8 -ns 8 -r 
python visual_scores.py -c configs/dip_vae_${DATASET}.yaml -d 10 -r 
python visual_scores.py -c configs/factorvae_${DATASET}.yaml -d 10 -r
python visual_scores.py -c configs/bbvae_${DATASET}.yaml -d 10 -r
python visual_scores.py -c configs/dip_vae_${DATASET}.yaml -d 128 -r 
python visual_scores.py -c configs/factorvae_${DATASET}.yaml -d 128 -r
python visual_scores.py -c configs/bbvae_${DATASET}.yaml -d 128 -r
```

In order to visualize the results and tune the angles of the TorusVAE:
```bash
python visualize.py -c configs/evae_${DATASET}_cosine.yaml -d 8 -ns 8 -r 
```

All results are saved to the logs/${DATASET} folder. Inside the checkpoint folder of each experiment there are yaml files containing the results.
* metrics_DCI_lasso.yaml - Disentanglement, Completeness and Informativeness
* metrics_RECONLOSS.yaml  - MSE
* metrics_visual.yaml - FID


# 2dshapes dataset generation
In order to generate the 2dshapes dataset, simply execute,
```bash
python generate_shapes.py
```


## Requirements:

python==3.7.10
pytorch==1.8.1
tqdm==4.60.0
ray==1.3.0
scipy==1.6.2
pytorch-lightning==1.2.6
numpy==1.19.2
matplotlib==3.3.4
scikit-image==0.17.2
scikit-learn==0.24.1
test-tube==0.7.5
cupy-cuda102==8.6.0
pytorch-fid==0.2.0
