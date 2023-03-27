import random
from datetime import datetime
import numpy as np

A = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [2, 2, 2, 2]]
B = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [2, 2, 2, 2]]


def ex1_u():
    m = 0
    u = pow(10, -m)
    while 1 + u != 1:
        m = m + 1
        u = pow(10, -m)
    return u


def ex1_not_u():
    m = 0
    not_u = pow(10, -m - 1)
    while 1 + not_u == 1:
        m = m + 1
        not_u = pow(10, -m - 1)
    return not_u


print('For u precision is', ex1_u())
print('For not u precision is', ex1_not_u())


def ex2a():
    x = 1.0
    y = ex1_not_u()
    z = ex1_not_u()
    if (x + y) + z != x + (y + z):
        return "Operatia + este neasociativa"


print(ex2a())


def ex2b():
    count = 0
    start_time = datetime.now()
    while True:
        count = count + 1
        x = random.random()
        y = random.random()
        z = random.random()
        if (x * y) * z != x * (y * z):
            print("Exemplu pentru o operatie de inmultire cu valorile", x, y, z, "este neasociativa")
            break
    end_time = datetime.now()
    print("The time of execution of above program is :", (end_time - start_time).total_seconds() * 10 ** 3, "ms")
    print(count)


ex2b()


def split(matrix):
    size_matrix = len(matrix) // 2
    a11 = [[0 for j in range(0, size_matrix)] for i in range(0, size_matrix)]
    a12 = [[0 for j in range(0, size_matrix)] for i in range(0, size_matrix)]
    a21 = [[0 for j in range(0, size_matrix)] for i in range(0, size_matrix)]
    a22 = [[0 for j in range(0, size_matrix)] for i in range(0, size_matrix)]
    for i in range(0, size_matrix):
        for j in range(0, size_matrix):
            a11[i][j] = matrix[i][j]
            a12[i][j] = matrix[i][j + size_matrix]
            a21[i][j] = matrix[i + size_matrix][j]
            a22[i][j] = matrix[i + size_matrix][j + size_matrix]
    return a11, a12, a21, a22


def strassen(A, B, q):
    # size of matrices is 1x1
    if len(A) < q:
        return np.dot(A, B)

    a11, a12, a21, a22 = np.array(split(A))
    b11, b12, b21, b22 = np.array(split(B))

    p1 = strassen(a11 + a22, b11 + b22, q)
    p2 = strassen(a21 + a22, b11, q)
    p3 = strassen(a11, b12 - b22, q)
    p4 = strassen(a22, b21 - b11, q)
    p5 = strassen(a11 + a12, b22, q)
    p6 = strassen(a21 - a11, b11 + b12, q)
    p7 = strassen(a12 - a22, b21 + b22, q)

    c11 = p1 + p4 - p5 + p7
    c12 = p3 + p5
    c21 = p2 + p4
    c22 = p1 + p3 - p2 + p6

    c = np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))

    return c


print(strassen(A, B, 2))
