## The script in this folder is used to simulate the PPower method on any generative model.
## The loaded model should be pre-trained and all its parameters should be frozen before running the projection operation.
## In our case, since our generative model had intermediate iterates, we optimize 3 different noise vectors, one at the last layer of the model, other at the end of the intermediate layer
## and the last at a reduced dimension.