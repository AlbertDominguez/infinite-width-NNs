import jax.numpy as np
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split

def split_data(features, target, seed=42, val_frac=0.15, test_frac=0.15):
    trX, testX, trY, testY = train_test_split(features, target, random_state=seed, test_size=test_frac)
    trX, valX, trY, valY = train_test_split(trX, trY, random_state=seed*2, test_size=test_frac)
    return trX, trY, valX, valY, testX, testY

def geodesic_error(target, preds):
    return np.mean(np.array([geodesic(t, p).km for t, p in zip(target, preds)]))

def mean_squared_error(target, preds):
    return np.mean((target-preds)**2)

def batch_generator(X, y, b_size):
    for i in range(0, len(X), b_size):
        yield X[i:i+b_size], y[i:i+b_size]