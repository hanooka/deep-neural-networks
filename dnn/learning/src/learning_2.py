import sklearn.datasets as skds
import matplotlib.pyplot as plt

# X, Y = skds.make_blobs(n_samples=100,n_features=2,
x, y = skds.make_blobs(n_samples=100, n_features=2, centers=2, random_state=1)

print(x[:5,:], type(x))
print(y[:5], type(y))



#plt.scatter(x[:, 0], x[:, 1], c=y, cmap="Greys", edgecolor="black")

#plt.show()

import numpy as np
z = np.linspace(-10,10,1000)
y = 1 / (1+np.exp(-z))
plt.plot(z,y)
plt.show()