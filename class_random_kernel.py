# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 18:05:43 2022

@author: GustavoSanchez
"""

# import basic package
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifierCV
from numba import njit, prange

@njit("Tuple((float64[:],int32[:],float64[:],int32[:],int32[:]))(int64,int64)")
def generate_kernels(input_length, num_kernels):

    candidate_lengths = np.array((7, 9, 11), dtype = np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels)

    weights = np.zeros(lengths.sum(), dtype = np.float64)
    biases = np.zeros(num_kernels, dtype = np.float64)
    dilations = np.zeros(num_kernels, dtype = np.int32)
    paddings = np.zeros(num_kernels, dtype = np.int32)

    a1 = 0

    for i in range(num_kernels):

        _length = lengths[i]

        _weights = np.random.normal(0, 1, _length)

        b1 = a1 + _length
        weights[a1:b1] = _weights - _weights.mean()

        biases[i] = np.random.uniform(-1, 1)

        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) / (_length - 1)))
        dilation = np.int32(dilation)
        dilations[i] = dilation

        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

        a1 = b1

    return weights, lengths, biases, dilations, paddings

@njit(fastmath = True)
def apply_kernel(X, weights, length, bias, dilation, padding):

    input_length = len(X)

    output_length = (input_length + (2 * padding)) - ((length - 1) * dilation)

    _ppv = 0
    _max = np.NINF

    end = (input_length + padding) - ((length - 1) * dilation)

    for i in range(-padding, end):

        _sum = bias

        index = i

        for j in range(length):

            if index > -1 and index < input_length:

                _sum = _sum + weights[j] * X[index]

            index = index + dilation

        if _sum > _max:
            _max = _sum

        if _sum > 0:
            _ppv += 1

    return _ppv / output_length, _max

@njit("float64[:,:](float64[:,:],Tuple((float64[::1],int32[:],float64[:],int32[:],int32[:])))", parallel = True, fastmath = True)
def apply_kernels(X, kernels):

    weights, lengths, biases, dilations, paddings = kernels

    num_examples, _ = X.shape
    num_kernels = len(lengths)

    _X = np.zeros((num_examples, num_kernels * 2), dtype = np.float64) # 2 features per kernel

    for i in prange(num_examples):

        a1 = 0 # for weights
        a2 = 0 # for features

        for j in range(num_kernels):

            b1 = a1 + lengths[j]
            b2 = a2 + 2

            _X[i, a2:b2] = \
            apply_kernel(X[i], weights[a1:b1], lengths[j], biases[j], dilations[j], paddings[j])

            a1 = b1
            a2 = b2

    return _X

np.random.seed(42)

def make_patterns_ndim(
    ndim, 
    number_of_datapoints,
):
    
    X = np.zeros([number_of_datapoints,ndim])
    S = np.zeros(number_of_datapoints*ndim)
    Y = np.zeros(number_of_datapoints)
    for i in range(number_of_datapoints):

        if np.random.uniform(-1,1) >= 0:
           q = 0.4*np.cos(np.arange(ndim)) + 0.6*np.random.uniform(-1,1,size=ndim)
           Y[i] = 0
        else:
           q = 0.4*np.cos(2*np.arange(ndim)) + 0.6*np.random.uniform(-1,1,size=ndim)
           Y[i] = 1
           
        S[i*ndim:(i+1)*ndim] = q

        X[i,:] = q

    return X, Y, S


X, y, S = make_patterns_ndim(
    ndim=32,
    number_of_datapoints=1000)

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

kernels = generate_kernels(X_train.shape[-1], 10_000)

X_training_transform = apply_kernels(X_train, kernels)
classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
classifier.fit(X_training_transform, y_train)

X_test_transform = apply_kernels(X_test, kernels)
print(classifier.score(X_test_transform, y_test))

