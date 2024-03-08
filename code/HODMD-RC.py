import scipy.sparse as sparse
import matplotlib.pyplot as plt
import numpy as np
import time
from numpy.lib.stride_tricks import as_strided
from pydmd import HODMD
from pydmd.plotter import plot_eigs,plot_eigs_mrdmd,plot_modes_2D,plot_snapshots_2D,plot_summary
import numpy as np

model_params = {'tau': 0.25,
                'nstep': 2500,
                'N': 200,
                'd': 200}

res_params = {'radius':0.3,
             'degree': 800,
             'sigma': 0.3,
             'Dr': 4000,
             'train_length': 8000,
             'predict_length': 2000,
            'num_inputs': model_params['N'],
             'beta': 0.00001
              }

def mean_relative_l2_error(x_true, x_pred):
    l2_norm_true = np.linalg.norm(x_true, axis=1)
    l2_norm_diff = np.linalg.norm(x_true - x_pred, axis=1)
    mean_relative_l2_error = np.mean(l2_norm_diff / l2_norm_true)
    return mean_relative_l2_error

# Randomly generated reservoir networks
def generate_reservoir(size,radius,degree):
    sparsity = degree/float(size);
    A = sparse.rand(size,size,density=sparsity).todense()
    vals = np.linalg.eigvals(A)
    e = np.max(np.abs(vals))
    A = (A/e) * radius
    return A

# The initial reservoir state space is activated using a nonlinear activation function. The procedure reads the training data to activate.
def reservoir_layer(A, Win, input, res_params):
    # Assigning values to the initial state space of the reservoir
    states = np.zeros((res_params['train_length'],res_params['Dr']))
    states[0] = np.zeros(res_params['Dr'])
    # Cyclic update of reservoir state space by training time
    for i in range(1, res_params['train_length']):
        states[i] = np.tanh(A.dot(states[i-1]) + Win.dot(input[i-1]) )
    # Retain the last moment state vector of the training time for subsequent predictions
    states_nearfuture = np.tanh( A.dot(states[res_params['train_length']-1]) + Win.dot(input[res_params['train_length']-1]) )
    return states,states_nearfuture

# Training reservoir phase
def train_reservoir(res_params, data):
    A = generate_reservoir(res_params['Dr'], res_params['radius'], res_params['degree'])
    q = int(res_params['Dr']/res_params['num_inputs'])
    Win = np.zeros((res_params['Dr'],res_params['num_inputs']))
    for i in range(res_params['num_inputs']):
        np.random.seed(seed=i)
        Win[i*q: (i+1)*q,i] = res_params['sigma'] * (-1 + 2 * np.random.rand(1,q)[0])
    states,states_nearfuture = reservoir_layer(A, Win, data, res_params)
    Wout = train(res_params, states, data)
    return states_nearfuture, Wout, A, Win

# Solve the weight matrix
def train(res_params,states,data):
    beta = res_params['beta']
    RC_states = np.hstack( (states, np.power(states, 2), np.power(states, 3) ,np.power(states, 4)) )
    Wout =(data[0:res_params['train_length']].T).dot(RC_states).dot (np.linalg.inv(RC_states.T.dot(RC_states) + beta * np.eye(4 * res_params['Dr'])))
    return Wout

# Forecasting phase
def predict(A, Win, res_params, states_nearfuture, Wout):
    output = np.zeros((res_params['predict_length'], res_params['Dr']))
    output_states = np.zeros((res_params['predict_length'], 4*res_params['Dr']))
    output[0] = states_nearfuture
    output_states[0] = np.hstack( (output[0], np.power(output[0], 2),np.power(output[0], 3),np.power(output[0], 4)) )
    # Autonomous forecasting phase
    for i in range(1,res_params['predict_length']):
        output[i] = np.tanh(A.dot(output[i-1]) + Win.dot( Wout.dot(output_states[i-1].T) ) )
        output_states[i] = np.hstack( (output[i], np.power(output[i], 2),np.power(output[i], 3),np.power(output[i], 4)) )
    predict = Wout.dot(output_states.T)
    return predict

train_data = np.load("./data/KS.npy")[0:8000]

hodmd = HODMD(      svd_rank=0,
                    tlsq_rank=0,
                    exact=True,
                    opt=True,
                    d=3).fit(train_data[None])
print("ERROR:")
result_ERROR=mean_relative_l2_error(train_data,hodmd.reconstructed_data.real)
print (result_ERROR)

states_nearfuture,Wout,A,Win = train_reservoir(res_params,hodmd.reconstructed_data.real)
output = predict(A, Win,res_params,states_nearfuture,Wout)
Predict = np.transpose(output)
