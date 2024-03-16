from sklearn.metrics import mean_absolute_error
import numpy as np
import pickle
import torch
import os
import lmdb
from tqdm import tqdm
import matplotlib.pyplot as plt
from ocpmodels.trainers import ForcesTrainer, BaseTrainer, BETrainer
from ocpmodels.datasets import SinglePointLmdbDataset
from ocpmodels import models
from ocpmodels.common import logger
from ocpmodels.common.utils import setup_logging

data_list = 'data.pkl'

with open(data_list, 'rb') as f:
    Data = pickle.load(f)
    
print('create LMDB database')
print('create data set')

db = lmdb.open(
    "data/test_set.lmdb",
    #map_size=10e1,
    subdir=False,
    meminit=False,
    map_async=True,
    map_size = (10**9) # 10 gb
)

idx = 0
for tmp in tqdm(Data[int(len(Data)*0.9):]):
    txn = db.begin(write=True)
    txn.put(f"{idx}".encode("ascii"), pickle.dumps(tmp, protocol=-1))
    txn.commit()
    db.sync()
    idx += 1
db.close()

task = {
    'dataset': 'single_point_lmdb', # dataset used for the S2EF task
    'description': 'Relaxed state energy prediction from initial structure.',
    'type': 'regression',
    'metric': 'mae',
    'labels': ['relaxed energy'],
    #'grad_input': 'atomic forces',
    #'train_on_free_atoms': True,
    #'eval_on_free_atoms': True
}

test_src = 'data/all_double_perv_te.lmdb'

# Dataset
dataset = [{'src': test_src}, # train set
  {'src': test_src}, # val_set
  {'src': test_src}
]

# Optimizer
optimizer = {
    'batch_size': 16, # if hitting GPU memory issues, lower this
    'eval_batch_size': 16,
    'load_balancing': 'atoms',
    'eval_every': 100,
    'num_workers': 0,
    'lr_initial': 5.e-5,
    'optimizer' : 'AdamW',
    'optimizer_params' : {'amsgrad':True},
    'scheduler': "ReduceLROnPlateau",
    'mode': "min",
    'factor': 0.8,
    'patience': 3,
    'max_epochs': 5000,
    'force_coefficient': 100,
    'energy_coefficient':1,
    'ema_decay': 0.999,
    'clip_grad_norm':10,
    'loss_energy': 'mae',
    'loss_force':'l2mae',
    'weight_decay':0,
}

model_params = {
    'name': 'gemnet_oc',
    'num_spherical': 7,
    'num_radial': 128,
    'num_blocks': 4,
    'emb_size_atom': 256,
    'emb_size_edge': 512,
    'emb_size_trip_in': 64,
    'emb_size_trip_out': 64,
    'emb_size_quad_in': 32,
    'emb_size_quad_out': 32,
    'emb_size_aint_in': 64,
    'emb_size_aint_out': 64,
    'emb_size_rbf': 16,
    'emb_size_cbf': 16,
    'emb_size_sbf': 32,
    'num_before_skip': 2,
    'num_after_skip': 2,
    'num_concat': 1,
    'num_atom': 3,
    'num_output_afteratom':3,
    'cutoff': 12.0,
    'cutoff_qint': 12.0,
    'cutoff_aeaint': 12.0,
    'cutoff_aint': 12.0,
    'max_neighbors': 30,
    'max_neighbors_qint': 8,
    'max_neighbors_aeaint': 20,
    'max_neighbors_aint': 1000,
    'rbf': {'name': 'gaussian'},
    'envelope': {'name': 'polynomial', 'exponent': 5},
    'cbf': {'name': 'spherical_harmonics'},
    'extensive': True,
    'output_init': 'HeOrthogonal',
    'activation': 'silu',
    'scale_file': 'configs/s2ef/all/gemnet/scaling_factors/gemnet-oc.pt',
    'regress_forces': False,
    'direct_forces': True,
    'forces_coupled':False,
    'quad_interaction':True,
    'atom_edge_interaction':True,
    'edge_atom_interaction':True,
    'atom_interaction':True,
    'num_atom_emb_layers':2,
    'num_global_out_layers':2,
    'qint_tags': [1,2],
}
  
pretrained_trainer = BETrainer(
    task=task,
    model=model_params,
    dataset=dataset,
    optimizer=optimizer,
    identifier="gemnet-oc-double_perov",
    run_dir="./", # directory to save results if is_debug=False. Prediction files are saved here so be careful not to override!
    is_debug=False, # if True, do not save checkpoint, logs, or results
    #is_vis=False,
    print_every=20,
    seed=0, # random seed to use
    logger="tensorboard", # logger of choice (tensorboard and wandb supported)
    local_rank=0,
)

checkpoint_path='checkpoint/best_checkpoint.pt'
pretrained_trainer.load_checkpoint(checkpoint_path=checkpoint_path)
predictions = pretrained_trainer.predict(pretrained_trainer.test_loader, results_file="predict_result", disable_tqdm=False)

