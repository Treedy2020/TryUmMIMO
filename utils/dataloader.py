import torch
import torch.nn as nn
import numpy as np
import h5py
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

class DataGenerator:
  def __init__(self, Nt, Nr, c, c_sub, number, max_value, data_root=None):
    self.Nt = Nt
    self.Nr = Nr
    self.c = c
    self.c_sub = c_sub
    self.number = number
    self.max_value = max_value
    self.data_root = data_root
    
  def generate(self):
    raw_data = self.max_value*np.random.uniform(-1, 1, (self.number, self.c_sub*self.c, self.Nt*self.Nr))
    return raw_data

  def fromDataRoot(self):
    mat_transpose = h5py.File(self.data_root)['Final'][:self.c_sub*self.number]
    mat = mat_transpose.T.reshape(self.Nt, self.Nr, self.c, self.c_sub, self.number)
    mat = mat.transpose(-1, 0, 1, 2, 3)
    
    mat_flat = mat.flatten()
    mat_real = np.array([mat_flat[i][0] for i in range(mat_flat.shape[0])])
    mat_img = np.array([mat_flat[i][1] for i in range(mat_flat.shape[0])])
    
    mat_real.resize((self.number, self.Nt, self.Nr, self.c, self.c_sub))
    mat_img.resize((self.number, self.Nt, self.Nr, self.c, self.c_sub))
    
    return mat_real

class MimoDataset(Dataset):
  def __init__(self, raw_data_array):
    self.raw_data_array = torch.tensor(raw_data_array, dtype=torch.float32)

  def __len__(self):
    return self.raw_data_array.shape[0]
  
  def __getitem__(self, ind):
    src, target = self.raw_data_array[ind], self.raw_data_array.clone()[ind]
    return src, target

class MimoDataLoader:
  def __init__(self, dataset, batch_size=2, train_rho=0.8, val_rho=0.1):
    self.dataset = dataset
    self.dataset_len = len(self.dataset)
    self.batch_size = batch_size
    self.train_num = int(train_rho*self.dataset_len)
    self.val_num = int(val_rho*self.dataset_len)
    self.test_num = self.dataset_len - self.train_num - self.val_num
    assert self.train_num and self.val_num and self.test_num, ValueError("The hyperparameter [number: {:n}] is too small for dataset building, try larger one.".format(self.dataset_len))
  
  def getDataLoader(self):
    train_dataset, val_dataset, test_dataset, = random_split(self.dataset, [self.train_num, self.val_num, self.test_num])
    
    train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
    
    return train_dataloader, test_dataloader, val_dataloader