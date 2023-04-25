import random

import numpy as np


def f(x):
    return x ** 2 - 12 * x + 30


def aitken(n, x, y):
    F = np.zeros((n, n))
    F[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            F[i, j] = (F[i + 1, j - 1] - F[i, j - 1]) / (x[i + j] - x[i])

    return F[0, :]


def ex1(n, x, y, t):
    dif = aitken(n, x, y)
    Ln = dif[0]

    for i in range(1, n):
        prod = dif[i]
        for j in range(i):
            prod *= (t - x[j])
        Ln += prod
    print("Ln(x):", Ln)
    print("|Ln(x) − f(x)|:", abs(Ln - f(t)))


def horner(c, t):
    Px = c[0]
    for i in range(1, len(c)):
        Px = Px * t + c[i]
    return Px


def ex2(n, x, y, t):
    m = 3
    c = np.polyfit(x, y, m)
    Pmx = horner(c, t)
    print("Pm(x):", Pmx)
    print("|Pm(x) − f(x)|:", abs(Pmx - f(t)))
    sum = 0
    for i in range(0, n):
        sum += abs(horner(c, x[i]) - y[i])
    print("Suma de la 0 la n din |Pm(xi) − yi|:", sum)


if __name__ == '__main__':
    n = 10
    a = 1
    b = 5
    x = np.random.uniform(1, 5, n + 1)
    x.sort()
    x[0] = 1
    x[n] = 5
    # print(x)
    y = np.zeros(n + 1)
    for i in range(0, n + 1):
        y[i] = f(x[i])
    # print(y)
    t = x[0] - x[1]
    ex1(n + 1, x, y, t)
    ex2(n + 1, x, y, t)
