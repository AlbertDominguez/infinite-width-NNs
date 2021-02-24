# Infinite-width neural networks

WIP!

The aim of this project was to study infinite-width neural networks from a practical point of view, including the performance of finite ensembles made up of different infinite networks, as well as performance comparison with finite-width counterparts. Only multilayer perceptron (MLPs) architectures have been studied. The associated paper will be made available here in the near future.

Currently, only the code base is available, which includes the possibility of training finite MLPs and compute kernel matrices of their infinite counterparts, while also computing a desired metric (currently, only the mean geodesic error (used in the paper) and the mean squared error are available). Some basic functionalities are still lacking, such as creation of needed directories if they do not exist. 

Evaluation of finite ensembles of both finite and infinite networks will be added in the future, as well as the possibility of using other non-linearities in the hidden layers (as of now, ReLUs are used).
