#Calculate both t and t/2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
import tqdm
import random
import matplotlib.pyplot as plt
import helpers
import unetlayerclass
import loss_function
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CelebA
import tqdm
import random
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder

## weight to be assigned to the loss from image at timestep t, from the last layer of U-net. // The weight for the image at timestep t/2 will be 1-loss_weight
HYPER_PARAMS = {'loss_weight' : 1}

## No. of epochs
n_epochs =   40
## size of a mini-batch
batch_size =   32
## learning rate
lr=1e-4 


device = 'cuda'
sigma =  5

marginal_prob_std_fn = functools.partial(helpers.marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(helpers.diffusion_coeff, sigma=sigma)

# Training
def train(search_params):
  
  score_model = torch.nn.DataParallel(unetlayerclass.ScoreNet(marginal_prob_std=marginal_prob_std_fn))
  score_model = score_model.to(device)
 
  dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
  print("current weight : ", search_params['loss_weight'])
  optimizer = Adam(score_model.parameters(), lr=lr)
  tqdm_epoch = tqdm.trange(n_epochs)
  for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0
    for x, y in data_loader:
      x = x.to(device)    
      
      loss = loss_function.loss_fn(score_model, x, marginal_prob_std_fn,search_params['loss_weight'])

      optimizer.zero_grad()
      loss.backward()    
      optimizer.step()
      avg_loss += loss.item() * x.shape[0]
      num_items += x.shape[0]
    # Print the averaged training loss so far.
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))


    torch.save(score_model.state_dict(), 'ckpt'+str(search_params['loss_weight'])+'.pth')
 
  return avg_loss / num_items

if __name__ == "__main__":
    train(HYPER_PARAMS)
