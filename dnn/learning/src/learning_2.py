import sklearn.datasets as skds
import torch
import matplotlib.pyplot as plt


def learning_2():
    # X, Y = skds.make_blobs(n_samples=100,n_features=2,
    X, Y = skds.make_blobs(n_samples=100, n_features=2, centers=2, random_state=1)

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    print(X[:5, :], type(X))
    print(Y[:5], type(Y))

    # plt.scatter(x[:, 0], x[:, 1], c=y, cmap="Greys", edgecolor="black")

    # plt.show()

    import numpy as np


    # z = np.linspace(-10,10,1000)
    # y = 1 / (1+np.exp(-z))
    # plt.plot(z,y)
    # plt.show()


    def draw_05_line():
        line = lambda x: -w0 / w1 * x - b / w1
        x0 = torch.tensor([-11, 0])
        x1 = line(x0)

        plt.plot(x0, x1)
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap="Greys", edgecolor="black")
        plt.show()


    w0, w1, b = 1, -1, 5.5
    draw_05_line()

    z = lambda w0, w1, b, x0, x1: w0 * x0 + w1 * x1 + b
    y = lambda z: 1 / (1 + torch.exp(-z))
    model = lambda w0, w1, b, x0, x1: y(z(w0, w1, b, x0, x1))


    def draw_prob_contours():
        x0, x1 = torch.linspace(-12, 1, 100), torch.linspace(-7, 7, 100)
        grid0, grid1 = torch.meshgrid(x0, x1)
        fig = plt.contour(grid0, grid1, model(w0, w1, b, grid0, grid1), cmap="Greys")
        fig.clabel(inline=True, fontsize=10)
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap="Greys", edgecolor="black")
        plt.show()


    draw_prob_contours()

    dHdy = lambda y, yt: -(yt - y) / (y - y ** 2)
    dydz = lambda y: y * (1 - y)
    dzdw0 = lambda x0: x0
    dzdw1 = lambda x1: x1
    dzdb = 1


    def calc_dC():
        dH = torch.zeros(len(Y), 3)
        for idx in range(len(Y)):
            data = (X[idx, 0], X[idx, 1], Y[idx])
            y_model = y(z(w0, w1, b, data[0], data[1]))

            A = dHdy(y_model, data[2])
            B = dydz(y_model)
            dH[idx, 0] = A * B * dzdw0(data[0])
            dH[idx, 1] = A * B * dzdw1(data[1])
            dH[idx, 2] = A * B * dzdb
        return dH.mean(0)


    dC = calc_dC()

    print(dC)

    w0, w1, b = 1, -1, 5.5

    alpha = 0.1
    dC = calc_dC()
    (w0, w1, b) = torch.tensor((w0, w1, b)) - alpha * dC
    print(w0, w1, b)
    draw_05_line()

    w0, w1, b = 0.1, -0.1, 0.2
    alpha = 0.8

    H = lambda y, yt: -(yt * torch.log(y) + (1 - yt) * torch.log(1 - y))
    cost_per_point = H(model(w0, w1, b, X[:, 0], X[:, 1]), Y)

    cost = torch.zeros(1000)
    for iter_num in range(len(cost)):
        dC = calc_dC()
        params = torch.tensor((w0, w1, b))
        (w0, w1, b) = params - alpha * dC
        cost_per_point = H(model(w0, w1, b, X[:, 0], X[:, 1]), Y)
        cost[iter_num] = cost_per_point.mean()

    draw_05_line()
    print(w0, w1, b)

    draw_prob_contours()

    plt.plot(range(len(cost)),cost)
    err = (cost[-1]-cost[-2])/cost[-2]
    dC = calc_dC()
    print(err, dC, sep='\n')
    plt.show()




def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def train_neuron(X, Y, output_dim = 1, epochs=100):
    n, input_dim = X.shape

    w0 = torch.randn((input_dim, output_dim))
    b = torch.randn((1, output_dim))

    for epoch in range(epochs):

        z1 = torch.matmul(X, w0) + b
        a1 = sigmoid(z1)

        # MSE
        loss = torch.mean( (a1 - Y.view(-1, 1)) **2 )
        dloss_da1 = 2*(a1-Y.view(-1, 1)) / n
        da_dz = sigmoid_derivative(z1)
        da_dw = X
        dz_db = 1

        delta = dloss_da1 * da_dz



def learning_2_active():
    X, Y = skds.make_blobs(n_samples=100, n_features=5, centers=2, random_state=1)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    train_neuron(X, Y)

if __name__ == '__main__':
    learning_2_active()
