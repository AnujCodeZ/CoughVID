import os
import torch
import numpy as np
from PIL import Image


class CoughVIDDataset(torch.utils.data.Dataset):
  def __init__(self, paths, transform=None):
    self.paths = paths
    self.transform = transform
  
  def __len__(self):
    return len(self.paths)
  
  def __getitem__(self, index):
    img = Image.open(self.paths[index])
    img = img.convert('RGB')
    img = self.transform(img)
    
    base_path = os.path.split(self.paths[index])[0]
    id_ = os.path.splitext(os.path.split(self.paths[index])[1])[0]
    label_name = base_path.split('/')[-1]
    
    mfcc = np.load(os.path.join('CoughVID/MFCC/', label_name, id_ + '.npy'))
    mfcc_padded = np.zeros((20, 500))
    mfcc_padded[:mfcc.shape[0], :mfcc.shape[1]] = mfcc[:, :500]
    mfcc = torch.Tensor(mfcc_padded)

    if label_name == 'covid':
      label = torch.Tensor([1.0])
    else:
      label = torch.Tensor([0.0])

    return (img, mfcc), label