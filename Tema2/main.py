import numpy as np
import scipy as scipy
import scipy.linalg


def ldlt(L, D, A, n):
    v = np.zeros((n, 1))
    eps = 10 ** -5
    for i in range(0, n):
        L[i, i] = 1
        for j in range(0, i):
            v[j] = L[i, j] * D[j]
        D[i] = A[i, i]
        for j in range(0, i):
            D[i] -= L[i, j] * v[j]
        for j in range(i + 1, n):
            L[j, i] = A[j, i]
            for k in range(0, i):
                L[j, i] -= L[j, k] * v[k]
            if abs(D[i]) > eps:
                L[j, i] /= D[i]
            else:
                print("nu se poate face impartirea")
    return L, D


def determinant(D, n):
    Dd=1
    for i in range(0,n):
        Dd = Dd * D[i]
    return Dd


def verify(A):
    if np.all(A) == np.all(np.transpose(A)) and np.linalg.det(A):
        return True
    else:
        return False


def subst(D, L, B, n):
    x = np.zeros((n, 1))
    z = np.zeros(((n, 1)))
    y = np.zeros((n, 1))
    for i in range(0, n):
        z[i] = B[i]
        for j in range(0, i):
            z[i] = z[i] - L[i, j] * z[j]
        y[i] = z[i] / D[i]
    print("z:", z)
    print("y:", y)
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        if i != n - 1:
            for j in range(n - 1, i, -1):
                x[i] = x[i] - L[j, i] * x[j]
    return x


if __name__ == '__main__':
    # A = np.array([[1, 2.5, 3], [2.5, 8.25, 15.5], [3, 15.5, 43]])
    # Punctul6:
    A = np.random.random_integers(0, 2000, size=(10, 10))
    A_symm = (A + A.T) / 2
    # print(A)
    n = A.shape[1]
    L = np.zeros((n, n))
    D = np.zeros((n, 1))
    if verify(A):
        L, D = ldlt(L, D, A, n)
        # L_T = np.transpose(L)
        # print("L:", L)
        print("D:", D)
        # print("L^T:", L_T)
        print("Determinantul lui A:", determinant(D, n))
    else:
        print("Matricea nu este simetrica/pozitiva")
    # B = np.array([[12], [38], [68]])
    B = np.random.random_integers(0, 2000, size=(10,1))
    x = subst(D, L, B, n)
    print("X:", x)

    # Punctul4:
    x = np.dot(np.linalg.inv(A), B)
    # P, L, U = scipy.linalg.lu(A)
    U = np.triu(A)
    L = np.tril(A)
    print("x:", x)
    print("L:", L)
    print("U:", U)
    # Punctul 5:
    eps = 10 ** -9
    Ainit = np.zeros(n)
    norma = 0
    for i in range(n):
        for j in range(n):
            Ainit[i] += A[i, j] * x[j]
        Ainit[i] -= B[i]
        norma += Ainit[i] ** 2
    norma = np.sqrt(norma)
    if norma < eps:
        print("Norma este:", norma)
    else:
        print("Norma nu este mai mica decat epsilon")
