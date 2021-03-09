from fastai.vision import *
import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
import pandas as pd
# Math
import math
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import scipy
import cv2

bs = 64
n_classes = 3
n_samples = 28

def tabulate(x, y, f):
    """Return a table of f(x, y). Useful for the Gram-like operations."""
    return np.vectorize(f)(*np.meshgrid(x, y, sparse=True))
def cos_sum(a, b):
    """To work with tabulate."""
    return(math.cos(a+b))

class GAF:

    def __init__(self):
        pass
    def __call__(self, series):
        """Compute the Gramian Angular Field of an image"""
        # Min-Max scaling
        min_ = np.amin(series)
        max_ = np.amax(series)
        scaled_series = (2*series - max_ - min_)/(max_ - min_)

        # Floating point inaccuracy!
        scaled_series = np.where(scaled_series >= 1., 1., scaled_series)
        scaled_series = np.where(scaled_series <= -1., -1., scaled_series)

        # Polar encoding
        phi = np.arccos(scaled_series)
        # Note! The computation of r is not necessary
        r = np.linspace(0, 1, len(scaled_series))

        # GAF Computation (every term of the matrix)
        gaf = tabulate(phi, phi, cos_sum)

        return(gaf, phi, r, scaled_series)

X = np.linspace(0,1,num=n_samples) **2
gaf = GAF()
g, _,_,_ = gaf(X)
print(g.shape)

plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.plot(X)
plt.title("Original", fontsize=16)
plt.subplot(122)
plt.imshow(g, cmap='rainbow', origin='lower')
plt.title("GAF", fontsize=16)
#plt.savefig('plog.svg')

xjo = pd.read_csv('./data/axjo.csv', parse_dates=['Date'], index_col=[0])
print('data read')
print(xjo.head().T)

xjo['Value'] = (xjo['High']+xjo['Low'])/2.0
y = xjo['Value']
print('\nvalue')
print(y.head().T)

ahead = y.shift(-1)
print ('\nshifted')
print(ahead.head().T)

print('\n y-1')
print((y-1).head().T)

new_y = 100.0*(ahead/y-1)
print('\nnew y')
print(new_y.head().T)

y = pd.qcut(new_y, 3,labels=False).dropna().astype('int')
print(y.head(10), xjo['Value'].head(10))
path = Path('./data')
for path in Path(path).iterdir():
    print(path)


