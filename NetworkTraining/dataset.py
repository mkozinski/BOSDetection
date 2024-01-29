import torch
from torch.utils.data import Dataset

class Dataset(Dataset):

  def __init__(self, img, lbl, augment):
    self.img=img
    self.lbl=lbl
    self.augment=augment

  def __len__(self):
    return len(self.lbl)

  def __getitem__(self, idx):
    i,l=self.augment(self.img[idx],self.lbl[idx])
    return  i,l

