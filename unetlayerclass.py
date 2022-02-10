import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import make_grid
import math
import copy

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]

class ScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""
    
  def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))
    # Encoding layers where the resolution decreases
    self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
    self.dense1 = Dense(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
    self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
    self.dense2 = Dense(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
    self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
    self.dense3 = Dense(embed_dim, channels[2])
    self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
    self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
    self.dense4 = Dense(embed_dim, channels[3])
    self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    

    #encoding layer for image at t/2 to directly be fed to layer 3 of U-Net
    self.convolve_12 = nn.Conv2d(1,64,5,stride=2,bias=False)

    #decoding layer for image at t/2 to resize it back to normal input image size, when outputted from de-conv layer 3(tconv3)
    self.tconv_12 = nn.ConvTranspose2d(64, 1, 6, stride=2, bias=False)

    # Decoding layers where the resolution increases
    self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
    self.dense5 = Dense(embed_dim, channels[2])
    self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
    self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)    
    self.dense6 = Dense(embed_dim, channels[1])
    self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
    self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)    
    self.dense7 = Dense(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
    self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)
    
    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)
    self.marginal_prob_std = marginal_prob_std
  

  def perturb_iterations_finder(self, x, z, random_t, eps=1e-5):  

    def cat_map(data,iteration):

      N = data.shape[1]

      # print("inside cat map")
      # create x and y components of Arnold's cat map
      x,y = np.meshgrid(range(N),range(N))
      
      xmap = (2*x+y) % N
      ymap = (x+y) % N
      
      xmap = torch.as_tensor(np.array(xmap).astype('long')).cuda()
      ymap = torch.as_tensor(np.array(ymap).astype('long')).cuda()

      new_data = torch.zeros_like(data)

      for i in range(iteration):
          new_data = data[:,xmap,ymap].clone()
      return new_data
        

    def map_iterator(data,random_t):

      # print("inside map iterator")
      copy_data = torch.zeros_like(data)
      for n in range(0,data.shape[0]):
        copy_data[n] = cat_map(data[n],random_t)
        
      return copy_data

    original = copy.deepcopy(x)

    perturbed_x = copy.deepcopy(x)

    N = 40

    dt = 1/N

    list_of_iterations = []
    
    list_of_iterations.append(x.clone())
    images_at_timestep = torch.zeros_like(x)
    images_at_half_timestep = torch.zeros_like(x)

    # print("iterations:",N)
    for i in range(N):
      # print("iteration step: "+ str(i+1) +"/",str(N))
      
      each_step = (i+1)/N

      random_normal = (5)**each_step * dt * z

      # cat_map_drift = dt * map_iterator(perturbed_x,1)
      perturbed_x +=  random_normal


      list_of_iterations.append(perturbed_x.clone())    

    index_final_image = 0

    for timestep in random_t * 40:

      if (np.round(timestep.item(),0) - timestep.item()) < 0:
        index_1 = int(np.round(timestep.item(),0))
        w1 = 1-(-(np.round(timestep.item(),0) - timestep.item()))

        tensor1 = list_of_iterations[index_1][index_final_image] * w1

        index_2 = int(np.round(timestep.item(),0)) + 1
        w2 = (-(np.round(timestep.item(),0) - timestep.item()))
        
        tensor2 = list_of_iterations[index_2][index_final_image] * w2

        pt_addition_result_ex = tensor1.add(tensor2)
        
        images_at_timestep[index_final_image] = pt_addition_result_ex.clone()
        index_final_image = index_final_image + 1
      else:
        index_1 = int(np.round(timestep.item(),0)-1)
        w1 = np.round(timestep.item(),0) - timestep.item()

        tensor1 = list_of_iterations[index_1][index_final_image] * w1

        index_2 = int(np.round(timestep.item(),0))
        w2 = 1-(np.round(timestep.item(),0) - timestep.item())

        tensor2 = list_of_iterations[index_2][index_final_image] * w2

        pt_addition_result_ex = tensor1.add(tensor2)

        images_at_timestep[index_final_image] = pt_addition_result_ex.clone()
        index_final_image = index_final_image + 1
    
    index_half_t_image = 0

    for timestep in random_t * 20:

      if (np.round(timestep.item(),0) - timestep.item()) < 0:
        index_1 = int(np.round(timestep.item(),0))
        w1 = 1-(-(np.round(timestep.item(),0) - timestep.item()))

        tensor1 = list_of_iterations[index_1][index_half_t_image] * w1

        index_2 = int(np.round(timestep.item(),0)) + 1
        w2 = (-(np.round(timestep.item(),0) - timestep.item()))
        
        tensor2 = list_of_iterations[index_2][index_half_t_image] * w2

        pt_addition_result_ex = tensor1.add(tensor2)
        
        images_at_half_timestep[index_half_t_image] = pt_addition_result_ex.clone()
        index_half_t_image = index_half_t_image + 1
      else:
        index_1 = int(np.round(timestep.item(),0)-1)
        w1 = np.round(timestep.item(),0) - timestep.item()

        tensor1 = list_of_iterations[index_1][index_half_t_image] * w1

        index_2 = int(np.round(timestep.item(),0))
        w2 = 1-(np.round(timestep.item(),0) - timestep.item())

        tensor2 = list_of_iterations[index_2][index_half_t_image] * w2

        pt_addition_result_ex = tensor1.add(tensor2)

        images_at_half_timestep[index_half_t_image] = pt_addition_result_ex.clone()
        index_half_t_image = index_half_t_image + 1


    return images_at_timestep,images_at_half_timestep


  def forward(self, x, z, t): 
    # Obtain the Gaussian random feature embedding for t   
    embed = self.act(self.embed(t))    
    # Encoding path
 
    image_t, image_t_by2 = self.perturb_iterations_finder(x,z,t)

    h1 = self.conv1(image_t)    
    ## Incorporate information from t
    h1 += self.dense1(embed)
    ## Group normalization
    h1 = self.gnorm1(h1)
    h1 = self.act(h1)
    h2 = self.conv2(h1)
    h2 += self.dense2(embed)
    h2 = self.gnorm2(h2)
    h2 = self.act(h2)
    h3 = self.conv3(h2)
    h3 += self.dense3(embed)
    h3 = self.gnorm3(h3)
    h3 = self.act(h3)
    h4 = self.conv4(h3)
    h4 += self.dense4(embed)
    h4 = self.gnorm4(h4)
    h4 = self.act(h4)

    embed_by2 = self.act(self.embed(t/2))  
    #encoding path for image at t/2
    t_by2_reduce_size = self.convolve_12(image_t_by2)

    h3_by2 = self.conv3(t_by2_reduce_size)
    h3_by2 += self.dense3(embed_by2)
    h3_by2 = self.gnorm3(h3_by2)
    h3_by2 = self.act(h3_by2)
    h4_by2 = self.conv4(h3_by2)
    h4_by2 += self.dense4(embed_by2)
    h4_by2 = self.gnorm4(h4_by2)
    h4_by2 = self.act(h4_by2)


    # Decoding path for image at t/2
    h_by2 = self.tconv4(h4_by2)

    ## Skip connection from the encoding path for t/2
    h_by2 += self.dense5(embed_by2)
    h_by2 = self.tgnorm4(h_by2)
    h_by2 = self.act(h_by2)
    h_by2 = self.tconv3(torch.cat([h_by2, h3_by2], dim=1))
    
    t_by_2_up_size = self.tconv_12(h_by2)

    # Decoding path
    h = self.tconv4(h4)

    ## Skip connection from the encoding path
    h += self.dense5(embed)
    h = self.tgnorm4(h)
    h = self.act(h)
    h = self.tconv3(torch.cat([h, h3], dim=1))

    
    h += self.dense6(embed)
    h = self.tgnorm3(h)
    h = self.act(h)
    h = self.tconv2(torch.cat([h, h2], dim=1))

  

    h += self.dense7(embed)
    h = self.tgnorm2(h)
    h = self.act(h)
    h = self.tconv1(torch.cat([h, h1], dim=1))

    
    # Normalize output
    h = h / self.marginal_prob_std(t)[:, None, None, None]
    t_by_2_up_size = t_by_2_up_size / self.marginal_prob_std(t/2)[:, None, None, None]
    
    return h,t_by_2_up_size
