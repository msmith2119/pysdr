
import numpy as np
import matplotlib.pyplot as plt

def resize_array(arr, N):

    M = len(arr)
    if M < N:
        # Zero-pad the array by creating a new view with zeros beyond M
        return np.pad(arr, (0, N - M), mode='constant')
    elif M > N:
        # Truncate the array by slicing (no new object)
        return arr[:N]
    else:
        # Return the original array if no change is needed
        return arr


def linear_convolution( x, h):
    N = len(x)
    M = len(h)
    y = np.zeros(N)  # Output size matches input size (clamped)

    for n in range(N):
        for m in range(M):
            if n - m >= 0:  # Only accumulate valid indices
                y[n] += x[n - m] * h[m]

    return y


def circular_convolution( x, hin, hwin):
    N = x.size  # Output size matches x
    y = np.zeros(N, dtype=np.float64)  # Initialize output array
    h = np.pad(hin, (0, N - hin.size), mode='constant')
    for n in range(N):
        for m in range(N):
            y[n] += x[m] * h[(n - m) % N]  # Circular indexing using modulo


    return y


def create_ola_function( M, N):
    assert M > N, "M must be greater than N"

    hanning_window = np.hanning(2 * N)

    def window_function(n):
        if n < N:
            return hanning_window[n]
        elif n < M - N:
            return 1.0
        else:
            return hanning_window[n - M + 2 * N]

    return window_function

def create_brick(N,m):
    z = np.ones(N)
    z[m:N-m] = 0
    return z
def create_notch(N,m,b):
    z = np.ones(N)
    z[m-b:m+b] = 0
    return z
def plot_array(w):
    N = len(w)  # Length of the array
    plt.figure(figsize=(8, 4))  # Set figure size
    plt.plot(np.arange(N), w, marker='o', linestyle='-')  # Plot with markers
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Plot of Array x")
    plt.grid(True)
