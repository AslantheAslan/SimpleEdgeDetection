import PIL
print('Pillow version:', PIL.__version__)

from d2l import mxnet as d2l
from mxnet import autograd, np, npx
from mxnet.gluon import nn

def corr2d(X, K): #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = np.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = d2l.reduce_sum((X[i: i + h, j: j + w] * K))
    return Y

from PIL import Image
import numpy as np
from numpy import asarray

X = np.ones((240, 320))*255
print(X)

X[:, 80:160] = 0
X[:, 240:300] = 0
K = np.array([[1.0, -1.0]])
print(X)
img = Image.fromarray(X)
img.show()
Y = corr2d(X, K)
print(Y)
vert = Image.fromarray(Y)
vert.show()