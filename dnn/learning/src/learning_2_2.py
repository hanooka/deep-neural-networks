import torch
import sklearn.datasets as skds

import matplotlib.pyplot as plt


def main():
    z = lambda w0, w1, b, x0, x1: w0*x0 + w1*x1 + b
    y = lambda z: 1/(1 + torch.exp(-z))
    model = lambda w0, w1, b, x0, x1: y(z(w0, w1, b, x0, x1))

    X, Y = skds.make_blobs(n_samples=100, n_features=2, centers=2, random_state=1)

    def draw_05_line():
        line = lambda x: -w0 / w1*x - b/w1
        x0 = torch.tensor([-11, 0])
        x1 = line(x0)
        plt.plot(x0, x1)
        plt.scatter(X[:, 0], X[:, 1],
                    c=Y, cmap="Greys", edgecolor="black")
        plt.show()

    w0, w1, b = 1, -1, 5.5
    draw_05_line()

if __name__ == '__main__':
    main()