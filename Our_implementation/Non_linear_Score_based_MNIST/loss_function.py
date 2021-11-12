import torch
import numpy as np

def loss_fn(model, x, marginal_prob_std, weight, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a 
      time-dependent score-based model.
    x: A mini-batch of training data.    
    marginal_prob_std: A function that gives the standard deviation of 
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """

  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
  
  z = torch.randn_like(x)

  std = marginal_prob_std(random_t)

  std2 = marginal_prob_std(random_t/2)

  image_t = torch.zeros_like(x)
  image_tby2 = torch.zeros_like(x)

  image_t,image_tby2 = model(x, z, random_t)

  # sample_grid2 = make_grid(image_t, nrow=int(np.sqrt(batch_size)))

  # plt.figure(figsize=(6,6))
  # plt.axis('off')
  # plt.imshow(sample_grid2.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
  # plt.show()
  # print("AT TIME T")


  # sample_grid2 = make_grid(image_tby2, nrow=int(np.sqrt(batch_size)))

  # plt.figure(figsize=(6,6))
  # plt.axis('off')
  # plt.imshow(sample_grid2.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
  # plt.show()
  # print("AT TIME T by 2")


  loss = weight*torch.mean(torch.sum((image_t * std[:, None, None, None] + z)**2, dim=(1,2,3))) + (1-weight) * torch.mean(torch.sum((image_tby2 * std2[:, None, None, None] + z)**2, dim=(1,2,3)))

  # loss = torch.mean(torch.sum((image_tby2 * std2[:, None, None, None] + z)**2, dim=(1,2,3)))

  return loss
