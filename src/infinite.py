import logging
import neural_tangents as nt
import pickle
import time

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def compute_and_save(batched_kernel_fn, trX, trY,
                     valX, valY, testX, testY,
                     filename, validation_metric_fn,
                     K_tr_tr=None, K_val_tr=None,
                     K_test_tr=None, dump=True):
    if K_tr_tr is None:
        log.info('Computing training kernel matrix...')
        t0_tr = time.time()
        K_tr_tr = batched_kernel_fn(trX, trX)
        te_tr = time.time()
        with open(f'./infinite/{filename}_K_train.pkl', 'wb') as fb:
            pickle.dump(K_tr_tr, fb)
    if K_val_tr is None:
        log.info('Computing validation kernel matrix...')
        t0_val = time.time()
        K_val_tr = batched_kernel_fn(valX, trX)
        te_val = time.time()
        with open(f'./infinite/{filename}_K_val.pkl', 'wb') as fb:
            pickle.dump(K_val_tr, fb)
    if K_test_tr is None:
        log.info('Computing test kernel matrix...')
        t0_test = time.time()
        K_test_tr = batched_kernel_fn(testX, trX)
        te_test = time.time()
        with open(f'./infinite/{filename}_K_test.pkl', 'wb') as fb:
            pickle.dump(K_test_tr, fb)
    log.info('Running inference...')
    predict_fn = nt.predict.gp_inference(K_tr_tr, trY, diag_reg=1e-4)
    t0_preds = time.time()
    val_preds_nngp = predict_fn(get='nngp', k_test_train = K_val_tr)
    te_val_nngp = time.time()
    test_preds_nngp = predict_fn(get='nngp', k_test_train = K_test_tr)
    te_test_nngp = time.time()
    val_preds_ntk = predict_fn(get='ntk', k_test_train = K_val_tr)
    te_val_ntk = time.time()
    test_preds_ntk = predict_fn(get='ntk', k_test_train = K_test_tr)
    te_test_ntk = time.time()
    val_score_nngp = validation_metric_fn(valY, val_preds_nngp)
    test_score_nngp = validation_metric_fn(testY, test_preds_nngp)
    val_score_ntk = validation_metric_fn(valY, val_preds_ntk)
    test_score_ntk = validation_metric_fn(testY, test_preds_ntk)
    log.info('[NNGP] Validation score is {}'.format(val_score_nngp))
    log.info('[NNGP] Test score is {}'.format(test_score_nngp))
    log.info('[NTK] Validation score is {}'.format(val_score_ntk))
    log.info('[NTK] Test score is {}'.format(test_score_ntk))
    results_obj = {
        'val_score_nngp': val_score_nngp,
        'test_score_nngp': test_score_nngp,
        'val_score_ntk': val_score_ntk,
        'test_score_ntk': test_score_ntk,
        'train_build_time': te_tr - t0_tr,
        'validation_build_time': te_val - t0_val,
        'test_build_time': te_test - t0_test,
        'val_inference_time_nngp': te_val_nngp - t0_preds,
        'test_inference_time_nngp': te_test_nngp - te_val_nngp,
        'val_inference_time_ntk': te_val_ntk - te_test_nngp,
        'test_inference_time_tnk': te_test_ntk - te_val_ntk
    }
    if dump:
        log.info('Inference finished. Dumping logs...')
        with open(f'./infinite/{filename}_infinite_logs.pkl', 'wb') as fb:
            pickle.dump(results_obj, fb)
    log.info('Done!')
    return results_obj, predict_fn, K_tr_tr, K_val_tr, K_test_tr