All the details about the experiments can be found in the paper: "IGO_dynamics_paper.pdf"


To run the code:

1) Setup the conda environment by using environment.yml 
2) Set the value of hyper parameter 'loss_weight' and Run main.py
3) To generate samples run the Sampler.py file after linking the location of the checkpoint of the trained model.


## Inside unetlayerclass.py the function "perturb_iterations_finder" is used to perturb the images 
with Arnold's Cat Map as the drift function of the SDE.

## The U-Net architecture was borrowed from https://github.com/yang-song/score_sde.


## The zipped file "Samples" contains the MNIST samples generated for different values of alpha.

