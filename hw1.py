import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np
import pandas as pd

def initialize_parameters(layer_dims):
    initialized_params = {}
    for i, _ in enumerate(layer_dims):
        if i+1 == len(layer_dims):
            break
        W_i = np.random.randn(layer_dims[i+1], layer_dims[i])
        b_i = np.zeros(layer_dims[i+1])
        initialized_params["W"+ str(i+1)] = W_i/layer_dims
        initialized_params["b" + str(i+1)] = b_i
    return initialized_params

#print (initialize_parameters([3*3, 4*4, 4]))

def linear_forward(A, W, b):
    Z = np.dot(W,A) + b
    return Z, {"W": W, "A":A, "b": b}

def softmax(Z):
    exp_Z = np.exp(Z)
    sum_exp_Z = sum(exp_Z)
    A = exp_Z/sum_exp_Z
    return A, Z

def relu(Z):
    if Z <= 0:
        A = 0
    else:
        A = Z
    return A, Z

def linear_activation_forward(A_prev, W, B, activation):
    (Z, linear_cache) = linear_forward(A_prev, W, B)
    if activation is 'relu':
        A, activation_cache = relu(Z)
    elif activation is 'softmax':
        A, activation_cache = softmax(Z)
    else:
        raise Exception('Only softmax or relu activations are permitted')
    joint_cache = linear_cache
    joint_cache["Z"] = activation_cache
    return joint_cache

def compute_cost(AL, Y):




