import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = 'cuda'
score_model = "######   LOAD YOUR GENERATIVE MODEL HERE   ######"


mnist_zeros_data = "CREATE MNIST (or any other) DATASET"

mnist_zeros_data = np.array(mnist_zeros_data)
mnist_zeros_data_norm = (mnist_zeros_data - mnist_zeros_data.mean())/(mnist_zeros_data.std())

#6742 in this case is the number of 0 digits in the MNIST trainset
std_X = mnist_zeros_data_norm.reshape(6742, 784)

V_batch = np.matmul(np.transpose(std_X), std_X)

diagonal_entries = {}
        
for i in range(V_batch.shape[0]):
    diagonal_entries[i] = V_batch[i][i]

sorted_diagonal_entries = dict(sorted(diagonal_entries.items(), key=lambda item: item[1]))

diagonal_list = list(sorted_diagonal_entries.keys())

w_0 = diagonal_list

initial_ws = []
for elements in w_0:
    initial_ws.append(V_batch[:][elements])

initial_ws = np.array(initial_ws)

def estimator(model, mean_image_last, mean_image_inter, timestep, iterations):

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    lossfunc = nn.MSELoss(reduction='sum')
    
    for i in range(iterations):

        #Here the model return the optimized noise vector parameter after each forward pass.
        predicted_image_last, predicted_image_inter = model(mean_image_last, mean_image_inter, timestep)

        #The noise vectors can be optimized one at a time or all together using a combinatorial loss.
        loss_last = lossfunc(predicted_image_last.to(device), mean_image_last.to(device))

        optimizer.zero_grad()
        loss_last.backward()    
        optimizer.step()

    return predicted_image_last, predicted_image_inter


outer_loop_iteration = 10

#number of updates to z by the model
inner_loop_iteration = 100


#here we have introduced two new parameters to the already pre-trained model.
#  module.starting_z is the noise vector which is optimized for generating images from the last layer.
#  module.starting_z_inter is the noise vector which is optimized for generating images from the intermediate layer.
for name, param in score_model.named_parameters():
    #FREEZE ALL PARAMETERS EXCEPT FOR THE NEWLY ADDED NOISE PARAMETERS
    if (name != 'module.starting_z') and (name != 'module.starting_z_inter'):
        param.requires_grad = False

#loop over all the components
for component_index in range(len(initial_ws)):
    w_inter = initial_ws[component_index]
    w_last = initial_ws[component_index]
    for j in range(outer_loop_iteration):

        update_w_last = np.matmul(V_batch, w_last)
        update_w_inter = np.matmul(V_batch, w_inter)

        #Project the components using the Generative model
        projected_w_last,projected_w_inter = estimator(score_model, (torch.from_numpy(update_w_last).reshape(1,1,28,28)).float(),(torch.from_numpy(update_w_inter).reshape(1,1,28,28)).float(), torch.ones(1, device=device, dtype=torch.float), inner_loop_iteration)
        

        w_last = projected_w_last.reshape(784,1).cpu().detach().numpy()
        w_inter = projected_w_inter.reshape(784,1).cpu().detach().numpy()


    #to be added for intermediate layers
    weights = np.matmul(np.matmul(np.transpose(projected_w_last.reshape(784,1).cpu().detach().numpy()), V_batch), projected_w_last.reshape(784,1).cpu().detach().numpy())

    V_batch = V_batch - np.matmul(projected_w_last.reshape(784,1).cpu().detach().numpy(), np.transpose(projected_w_last.reshape(784,1).cpu().detach().numpy())) * weights
    
    #save the projections
    torch.save(projected_w_last.reshape(784,1), 'projw_component_' + str(component_index) + '.pt') 
    #save the weights of the projections
    torch.save(weights, 'weight_component_' + str(component_index) + '.pt')
