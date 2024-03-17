# Gemnet-OC_for_double_provskite

# Installation

- need to install 'ocp' (https://github.com/Open-Catalyst-Project/ocp/tree/main)
- install the 'ocp' package with 'pip install -e .'.

# Guide

- Use the data_preprocessing.py to relax your structures with Gemnet-OC2022-IS2RE. The following line in data_preprocessing.py can be changed to change your data and the model.
```
data_list = 'Enter_your_data_list' : enter your data list for relaxation (['name1', 'name2',...] # initial structure)
checkpoint = 'Enter_your_checkpoint_for_relaxation.pt'
```

- Run the run_OCP.py to obtain binding energy with the transferred model. This code will make lmdb data from the relaxed structure and predict binding energy. The following lines can be changed to edit the data set and model.
```
data_list = 'data.pkl' : enter your dataset

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
```

- Set checkpoint
```
checkpoint_path = 'enter_your_checkpoint' # checkpoitn file name
pretrained_trainer.load_checkpoint(checkpoint_path=checkpoint_path)
predictions = pretrained_trainer.predict(pretrained_trainer.test_loader, results_file="predict_result", disable_tqdm=False)
```
