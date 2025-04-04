
import numpy as np

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
