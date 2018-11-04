from cmath import exp, pi
import numpy as np

#https://rosettacode.org/wiki/Fast_Fourier_transform#Python
def fft(x):
    N = len(x)
    if N <= 1: return x
    even = fft(x[0::2])
    odd =  fft(x[1::2])
    T= [exp(-2j*pi*k/N)*odd[k] for k in range(N//2)]
    return [even[k] + T[k] for k in range(N//2)] + \
           [even[k] - T[k] for k in range(N//2)]


#author: matias
def fftshift(x):
    N = len(x)
    par = True if len(x)%2 == 0 else False

    if par:
        half_size = N//2
        left = x[half_size:]
        right = x[:half_size]
    else:
        half_size = N//2 + 1
        left = x[half_size:]
        right = x[:half_size]

    return np.append(left, right)

print(' '.join("%5.3f" % abs(f)
               for f in fft([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])))


