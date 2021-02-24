import argparse
import logging
import jax.numpy as np
import neural_tangents as nt
import os
import pickle
import src.utils as utils
import sys
from jax import random
from src.finite import train_and_save
from src.infinite import compute_and_save
from src.architecture import define

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

if not os.path.exists('./finite'):
    os.makedirs('./finite')
if not os.path.exists('./infinite'):
    os.makedirs('./infinite')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, required=False, default=0.1,
                        help='Learning rate for finite-width training.')
    parser.add_argument('--batch_size', type=int, required=False, default=64,
                        help='Batch size for finite-width training and batch kernel inference.')
    parser.add_argument('--num_epochs', type=int, required=False, default=10,
                        help='Batch size for finite-width training and batch kernel inference.')
    parser.add_argument('--early_stopping_epochs', type=int, required=False, default=0,
                        help='If validation loss does not improve after this number of epochs, stop training. If 0, no early stopping is performed.')
    parser.add_argument('--early_stopping_tolerance', type=float, required=False, default=1e-5,
                        help='Tolerance of loss improvement in early stopping.')
    parser.add_argument('--num_hidden_layers', type=int, required=True,
                        help='Number of hidden FC layers in network.')
    parser.add_argument('--W_std', type=float, required=True,
                        help='Standard deviation of weights at initialization.')
    parser.add_argument('--b_std', type=float, required=True,
                        help='Standard deviation of biases at initialization.')
    parser.add_argument('--filename', type=str, required=True,
                        help='Base filename to store objects and logs.')
    parser.add_argument('--seed', type=int, required=False, default=4242,
                        help='Seed.')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to file containing data.')
    parser.add_argument('--hidden_neurons', type=int, required=False, default=512,
                        help='Number of neurons in hidden layers.')
    parser.add_argument('--output_dim', type=int, required=True,
                        help='Number of neurons in output layer.')
    parser.add_argument('--val_frac', type=float, required=True,
                        help='Percentage of samples to use as validation data.')
    parser.add_argument('--test_frac', type=float, required=True,
                        help='Percentage of samples to use as test data.')
    parser.add_argument('--metric_fn', type=str, required=True, choices=['mse', 'MSE', 'geodesic'],
                        help='Chosen metric to output score (not optimize!)')
    args = parser.parse_args()
    log.info(args)
    return args

def read_data(args):
    path = args.data_path
    if path.endswith('.pkl') or path.endswith('.pickle'):
        with open(path, 'rb') as fb:
            data = pickle.load(fb)
    else:
        log.error('Unrecognized data format. Make sure its in pickle format.')
        return [None]*6
    trX, trY, valX, valY, testX, testY = utils.split_data(
        np.array(data['features']), np.array(data['target']), seed=args.seed,
        val_frac=args.val_frac, test_frac=args.test_frac
    )
    return trX, trY, valX, valY, testX, testY

def main():
    args = parse_args()
    if args.metric_fn == 'geodesic':
        validation_metric_fn = utils.geodesic_error
    elif args.metric_fn in ['mse', 'MSE']:
        validation_metric_fn = utils.mean_squared_error
    key = random.PRNGKey(args.seed)
    log.info('Reading and preparing data...')
    trX, trY, valX, valY, testX, testY = read_data(args)
    if trX is None:
        return 1
    log.info('Defining model functions...')
    init_fn, apply_fn, _, batched_kernel_fn = define(args)
    log.info('Training finite network...')
    train_and_save(args.num_epochs, args.batch_size,
                   args.learning_rate, init_fn,
                   apply_fn, key, args.filename,
                   args.early_stopping_epochs,
                   args.early_stopping_tolerance,
                   trX, trY, valX, valY, testX, testY,
                   validation_metric_fn,
                   output_logs=True, params=None, dump=True)
    log.info('Computing infinite kernels...')
    compute_and_save(batched_kernel_fn, trX, trY, valX, valY,
                     testX, testY, args.filename,
                     validation_metric_fn, dump=True)
    return 0

if __name__ == '__main__':
    sys.exit(main())