import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(1)

"""
You will be implementing the building blocks of a convolutional neural network! 
Each function you will implement will have detailed instructions that will walk you through the steps needed:

Convolution functions, including:
    Zero Padding
    Convolve window
    Convolution forward
    Convolution backward (optional)
Pooling functions, including:
    Pooling forward
    Create mask
    Distribute value
    Pooling backward (optional)
"""

def zero_pad(X, pad):
    """Implement the following function, which pads all the images of a batch of examples X with zeros. Use np.pad().
    if want to pad the array "a" of shape $(5,5,5,5,5)$ with pad=1 for the 2nd dim, pad=3 for the 4th dim and
    pad=0 for the rest, you would do:
    a = np.pad(a, ((0,0), (1,1), (0,0), (3,3), (0,0)), 'constant', constant_values = (..,..))"""
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant')  # None
    return X_pad


""" Single step of convolution """
def conv_single_step(a_slice_prev, W, b):
    """    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    Returns:
    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    """
    s = a_slice_prev * W  # Element-wise product between a_slice and W
    Z = np.sum(s, axis=None)
    Z = Z + float(b)  # add bias term
    return Z

""" Convolutional Neural Networks - Forward pass 
In the forward pass, you will take many filters and convolve them on the input. 
Each 'convolution' gives you a 2D matrix output. 
You will then stack these outputs to get a 3D volume:
"""
def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function

    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"

    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape  # retrieve dims from A_prev
    (f, f, n_C_prev, n_C) = W.shape  # Retrieve dimensions from W's shape
    stride = hparameters.get('stride')
    pad = hparameters.get('pad')

    # Compute the dimensions of the CONV output volume
    n_h = np.floor(np.divide(n_H_prev - f + (2*pad), stride) + 1)
    n_w = np.floor(np.divide(n_W_prev - f + (2 * pad), stride) + 1)

    # Initialize the output volume Z with zero
    Z = np.zeros((m, n_h, n_w, n_C))
    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(X=A_prev, pad=pad)

    # loop over batch of training examples
    for i in range(m):
        a_prev_pad = A_prev_pad[i]  # Select ith training example's padded activation
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_C):
                    # Find the corners of the current "slice"
                    v_start = h * stride  # start position depends on the stride value
                    v_end = v_start + f  # f is length/width of (kernel / filter)
                    h_start = w * stride
                    h_end = h_start + f

                    # Use the corners to define the (3D) slice of a_prev_pad
                    a_slice_prev = a_prev_pad[v_start:v_end, h_start:h_end, :]  # row, col, depth
                    # Convolve (3D) slice with the correct filter W and bias b, to get back one output neuron.
                    Z[i, h, w, c] = conv_single_step(a_slice_prev=a_slice_prev, W=W[:,:,:,c], b=b[:,:,:,c])
    # Making sure your output shape is correct
    assert (Z.shape == (m, n_h, n_w, n_C))
    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)
    return Z, cache


"""Forward Pooling
Now, you are going to implement MAX-POOL and AVG-POOL, in the same function."""
def pool_forward(A_prev, hparameters, mode = "max"):
    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    stride = hparameters.get('stride')
    f = hparameters.get('f')

    # Define the dimensions of the output
    n_h = np.floor(np.divide(n_H_prev - f, stride))
    n_w = np.floor(np.divide(n_W_prev - f, stride))
    n_c = n_C_prev

    # Initialize output matrix A
    A = np.zeros((m, n_h, n_w, n_c))

    for i in range(m):
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):
                    v_start = h*stride
                    v_end = v_start + f
                    h_start = w * stride
                    h_end = h_start + f

                    a_prev_slice = A_prev[i, v_start:v_end, h_start:h_end, c]
                    # Compute the pooling operation on the slice
                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_prev_slice,axis=None)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice, axis=None)
    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)

    # Making sure your output shape is correct
    assert (A.shape == (m, n_h, n_w, n_c))
    return A, cache