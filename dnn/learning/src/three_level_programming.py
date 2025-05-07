from functools import reduce
from statistics import correlation, linear_regression
from random import normalvariate


def sstt():
    def indipendent_variable(mean=0.0, stdev=1.0, n=10_000):
        return [normalvariate(mean, stdev) for i in range(n)]

    def caused_by(*variables):
        return [sum(column) for column in zip(*variables, strict=True)]

    def controlled_for(Y, X):
        m, b = linear_regression(X, Y)
        return [y - m*x + b for x,y in zip(X,Y)]

    A = indipendent_variable()
    B = indipendent_variable()
    print(correlation(A, B))

    A = indipendent_variable()
    B = caused_by(A, indipendent_variable())
    print(correlation(A, B))


    C = indipendent_variable()
    A = caused_by(C, indipendent_variable())
    B = caused_by(C, indipendent_variable())
    print(correlation(A, B))


    C = indipendent_variable()
    A = caused_by(C, indipendent_variable())
    B = caused_by(C, indipendent_variable())
    X = controlled_for(B, C)
    Y = controlled_for(A, C)
    print(correlation(X, Y))


sstt()
import numpy as np
# print(list(range(5)))
# a = reduce(lambda x, y: x*y, [1, 2, 3, 4, 5])
# print(a)