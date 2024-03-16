# Gemnet-OC_for_double_provskite

# Installation

- need to install 'ocp' 

https://github.com/Open-Catalyst-Project/ocp/tree/main

# Guide

- make Graph data and your lmdb dataset
```
tmp = AtomToData()
tmp.S2EF(Data_list) # Data_list is list of ase Atoms : [[Atoms1, Ad_energy], [Atoms2, Ad_energy] ... ]
tmp.MakeLmdb(name='data', ratio_list=None) # name : file name, default of ratio_list is [0.8, 0.1, 0.1]
```

- dataset imformation for train and validation and predict
```
train_src = 'data/name_train.lmdb'
val_src = 'data/name_validataion.lmdb'
test_src = 'data/name_test.lmdb'

dataset = [{'src':train_src},
            {'src':val_src},
            {'src':test_src} # recommand do not input test_src for train -> #{'src':test_src}
          ]
```

- predict GemnetOC
```
checkpoint_path = '~~~' # checkpoitn file name
pretrained_trainer.load_checkpoint(checkpoint_path=checkpoint_path)
predictions = pretrained_trainer.predict(pretrained_trainer.test_loader, results_file="predict_result", disable_tqdm=False)
```

- You can obtain energy prediction values by running run_ocp.py.
