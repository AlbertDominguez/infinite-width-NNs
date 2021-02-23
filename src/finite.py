import gc
import logging
import jax.numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from jax.api import jit, grad, vmap
from jax.experimental import optimizers
from src.utils import batch_generator

def _train_network(num_epochs, batch_size, key, trX, trY, valX, valY,
                  init_fn, apply_fn, opt_init, opt_update,
                  get_params, plot_losses=True, params=None,
                  early_stopping_epochs=0, early_stopping_tol=1e-2,
                  output_logs=True):
    if params is None:
        _, params = init_fn(key, (-1, trX.shape[1]))
        if output_logs:
            logging.info('Network parameters randomly initialized.')
    elif output_logs:
        logging.info('Network parameters initialized from a trained model.')
    opt_state = opt_init(params)
    train_losses, val_losses = [None]*num_epochs, [None]*num_epochs
    consecutive_epochs_without_improvement = 0
    min_val_loss = 1e9
    loss = jit(lambda params, x, y: 0.5 * np.mean((apply_fn(params, x) - y) ** 2)) # MSE
    grad_loss = jit(lambda state, x, y: grad(loss)(get_params(state), x, y)) # MSE Derivative
    for i in range(num_epochs):
        gc.collect()
        if output_logs:
            logging.info(f'---------- EPOCH {i+1} ----------')
        t0 = time.time()
        for ctr, (X, y) in enumerate(batch_generator(trX, trY, batch_size)):
            opt_state = opt_update(i, grad_loss(opt_state, trX, trY), opt_state)
            curr_params = get_params(opt_state)
        train_losses[i] = loss(curr_params, trX, trY)
        val_losses[i] = loss(curr_params, valX, valY)
        if output_logs:
            logging.info(f'Training loss: {train_losses[i]}')
            logging.info(f'Validation loss: {val_losses[i]}')
        if i > 0 and early_stopping_epochs > 0:
            if val_losses[i]+early_stopping_tol > min_val_loss:
                consecutive_epochs_without_improvement += 1
            else:
                if val_losses[i] < min_val_loss:
                    min_val_loss = val_losses[i]
                consecutive_epochs_without_improvement = 0
            if consecutive_epochs_without_improvement == early_stopping_epochs:
                logging.info(f'Validation loss has stopped improving. Early stopping after epoch {i+1}')
                train_losses = train_losses[:i+1]
                val_losses = val_losses[:i+1]
                break
    logging.info('Training finished!')
    if plot_losses:
        plt.close()
        plt.figure()
        plt.plot(range(1, len(train_losses)+1), train_losses, c='blue', label='Training')
        plt.plot(range(1, len(val_losses)+1), val_losses, c='red', label='Validation')
        plt.title('Loss evolution during training')
        plt.legend()
        plt.show()
        logging.info('Lowest validation loss is {} at epoch #{}'.format(np.min(np.array(val_losses)), np.argmin(np.array(val_losses))))
    return curr_params, train_losses, val_losses

def train_and_save(num_epochs, batch_size,
                   learning_rate,
                   init_fn, apply_fn, key,
                   filename, early_stopping_epochs,
                   early_stopping_tol,
                   trX, trY, valX,
                   valY, testX, testY,
                   validation_metric_fn,
                   output_logs=False, params=None,
                   dump=False):
    assert filename is not None
    opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
    t0_tr = time.time()
    params, train_losses, val_losses = _train_network(num_epochs, batch_size, key,
                                                 trX, trY,
                                                 valX, valY,
                                                 init_fn, apply_fn,
                                                 opt_init, opt_update, get_params,
                                                 early_stopping_epochs=early_stopping_epochs,
                                                 early_stopping_tol=early_stopping_tol,
                                                 params=params,
                                                 plot_losses=False, output_logs=output_logs)
    te_tr = time.time()
    val_preds = apply_fn(params, valX)
    te_val = time.time()
    test_preds = apply_fn(params, testX)
    te_test = time.time()
    logging.info('Dumping logs and trained parameters...')
    to_save_obj = {
        'params': params,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_score': validation_metric_fn(valY, val_preds),
        'test_score': validation_metric_fn(testY, test_preds),
        'training_time': te_tr-t0_tr,
        'val_inference_time': te_val-te_tr,
        'test_inference_time': te_test-te_val
    }
    logging.info('Validation score is {}'.format(to_save_obj['val_score']))
    logging.info('Test score is {}'.format(to_save_obj['test_score']))
    if dump:
        logging.info('Dumping result...')
        with open(f'./finite/{filename}.pkl', 'wb') as fb:
            pickle.dump(to_save_obj, fb)
        logging.info('Done!')
    return to_save_obj