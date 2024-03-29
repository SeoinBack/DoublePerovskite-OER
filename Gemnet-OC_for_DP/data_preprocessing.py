from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.datasets import SinglePointLmdbDataset
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
import pickle
import pandas as pd
import os
from ase.io import read
from ase.optimize import BFGS
from ase import Atoms, Atom
from tqdm import tqdm
import gc
import numpy as np
from ase.visualize import view
import torch
from math import sqrt

class traj(object):
    def __init__(self, name):
        self.a2g = AtomsToGraphs(
                    max_neigh=50,
                    radius=6,
                    r_energy=False,
                    r_forces=False,
                    r_distances=True,
                    r_edges=True,
                    r_fixed=True)
        self.name = name
        self.data_list = []
        
    def S2EF(self, calc): ## if DFT data -> S2EF(self, ad_E, calc)
        if not os.path.exists(os.path.join('traj_file/', self.name)):
            #images = [self.i_xdata_f]
            # loading init structure
            img_tag = self.i_xdata_i
            
            # reset tags
            for i in range(len(img_tag.get_tags())):
                img_tag[i].tag = 0
                
            # find first layer
            atom_positions = img_tag.get_positions()
            Atoms_height = np.unique(np.sort(atom_positions[:,2]))
            first_layer = np.where(atom_positions == Atoms_height[4])
            
            # set tags surface to 1
            for surface in first_layer[0]:
                img_tag[surface].tag = 1
                
            # set adsorbate tag
            ad_height = Atoms_height[5:]
            for i in range(len(ad_height)):
                ad = np.where(atom_positions == ad_height[i])
                img_tag[ad[0][0]].tag = 2
              
            #relaxation - ml_relaxation using cuda:0
            img_tag.calc = calc
            image = BFGS(img_tag, trajectory='traj_file/' + self.name)
            
            image.run(fmax=0.05, steps=100)
            
            i_xdata = read(os.path.join('traj_file/', self.name))
            forces = i_xdata.get_forces()
            fmax = sqrt((forces**2).sum(axis=1).max())
            
            if fmax > 0.05:
                self.data_list.append(self.name)
            else:
                images = [i_xdata]
                
                data_objects = self.a2g.convert_all(images, disable_tqdm=True)
                
                tags = img_tag.get_tags()
                
                for fid, data in enumerate(data_objects):
                    data.sid = torch.LongTensor([0])
                    data.fid = torch.LongTensor(fid)
                    data.tags = torch.LongTensor(tags)
                    # data.y = calc.get_energy(i_xdata) # if you want to get energy
                
                return data
    
    def check_not_relax_data(self, calc):
        img_tag = read(os.path.join('traj_file/', self.name))
        
        #relaxation - ml_relaxation using cuda:0
        #print(img_tag.get_forces())
        forces = img_tag.get_forces()
        fmax = sqrt((forces**2).sum(axis=1).max())
        if fmax > 0.05:
            os.remove(os.path.join('traj_file/', self.name))
        
    def get_item(self):
        return self.data_list
    
def ParalIter(processes=32, maxtasksperchild=1):
    gc.collect()
    if 'get_ipython' in locals().keys(): # it doesnt work in ipython
        multiprocessing = None
    elif processes == 1:
        mapper = map
    else:
        try:
            from multiprocessing import Pool
            p = Pool(processes=processes, maxtasksperchild=maxtasksperchild)
            mapper = p.imap_unordered
        except:
            mapper = map

    return mapper

def ParalProcess(func,inputs, processes=32, maxtasksperchild=1):
    mapper = ParalIter(processes, maxtasksperchild)
    return list(tqdm(mapper(func,inputs),total = len(inputs)))

def get_data(name):
    data_class = traj(name)
    b = data_class.S2EF(calc)
    return b

if __name__ == '__main__':
    
    #%% using csv data
    data_list = 'Enter_your_data_list' # ['name1', 'name2', ...]
    
     #%% run relaxtion
    
    checkpoint = 'Enter_your_checkpoint_for_relaxation.pt'
    calc = OCPCalculator(checkpoint=checkpoint, cpu=False)
    
    relaxed_data = ParalProcess(get_data, data_list)

    pickle.dump(relaxed_data, open('data.pkl', 'wb'))