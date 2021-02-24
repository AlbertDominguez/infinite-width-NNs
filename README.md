# Infinite-width neural networks

WIP!

The aim of this project was to study infinite-width neural networks from a practical point of view, including the performance of finite ensembles made up of different infinite networks, as well as performance comparison with finite-width counterparts. Only multilayer perceptron (MLPs) architectures have been studied. The associated paper will be made available here in the near future.

Currently, only the code base is available, which includes the possibility of training finite MLPs and compute kernel matrices of their infinite counterparts, while also computing a desired metric (currently, only the mean geodesic error (used in the paper) and the mean squared error are available). Data must be read from a pickle object containing a dictionary of keys `features` and `target`, each containing (potentially multidimensional) arrays. Compatibility with more general data formats is yet to be added.

Evaluation of finite ensembles of both finite and infinite networks will be added in the future, as well as the possibility of using other non-linearities in the hidden layers (as of now, ReLUs are used).


## Instructions

Here is an example of the usage of the CLI.

```{bash}
$ python3 main.py --num_hidden_layers 1 --W_std 1 --b_std 0.05 --filename test_out_1 --data_path ./data/data.pkl --output_dim 2 --val_frac 0.15 --test_frac 0.15 --metric_fn MSE
```

Upon launch, a couple of folders named `finite` and `infinite` will be created if they do not exist. The former folder will be used to store parameters and metrics from finite network trainings (all stored in the same pickle object), while the latter will be used to store kernel matrices and metrics corresponding to infinite networks. Note that files may be overwritten if a specified filename has already been used before.

For more information about the arguments and its possible values, please run `python3 main.py -h`.
