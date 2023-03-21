import numpy as np

eps = 10 ** -6


def ex1(A, s, n):
    b = np.zeros((n, 1))
    for i in range(0, n):
        for j in range(0, n):
            b[i] = b[i] + s[j] * A[i, j]
    return b


def ex2(A, b, n):
    In = np.ones((n))
    q = np.zeros((n, n))
    for i in range(0, n):
        q[i, i] = 1
    for r in range(0, n - 1):
        sigma = 0
        for i in range(r, n):
            sigma = sigma + A[i, r] ** 2
        if sigma <= eps:
            break
        k = np.sqrt(sigma)
        if A[r, r] > 0:
            k = -k
        beta = sigma - k * A[r, r]
        u = np.zeros((n, 1))
        u[r] = A[r, r] - k
        for i in range(r + 1, n):
            u[i] = A[i, r]
        for j in range(r + 1, n):
            sum = 0
            for i in range(r, n):
                sum = sum + u[i] * A[i, j]
            teta = sum / beta
            for i in range(r, n):
                A[i, j] = A[i, j] - teta * u[i]
        A[r, r] = k
        for i in range(r + 1, n):
            A[i, r] = 0
        sum = 0
        for i in range(r, n):
            sum = sum + u[i] * b[i]
        teta = sum / beta
        for i in range(r, n):
            b[i] = b[i] - teta * u[i]
        for j in range(0, n):
            sum = 0
            for i in range(r, n):
                sum = sum + u[i] * q[i, j]
            teta = sum / beta
            for i in range(r, n):
                q[i, j] = q[i, j] - teta * u[i]
    # return np.linalg.inv(q)
    return q


def ex3QR(Ainit, b):
    q, r = np.linalg.qr(Ainit)
    print(q)
    print(r)
    x = np.zeros((n, 1))
    z = np.zeros(((n, 1)))
    for i in range(0, n):
        z[i] = b[i]
        for j in range(0, i):
            z[i] = z[i] - r[i, j] * z[j]
    for i in range(n - 1, -1, -1):
        x[i] = z[i]
        x[n - 1] = z[n - 1] / r[n - 1, n - 1]
        if i != n - 1:
            for j in range(n - 1, i, -1):
                x[i] = (x[i] - r[i, j] * x[j])
    return x


def ex3Householder(A, b):
    x = np.zeros((n, 1))
    z = np.zeros(((n, 1)))
    for i in range(0, n):
        z[i] = b[i]
        for j in range(0, i):
            z[i] = z[i] - A[i, j] * z[j]
    for i in range(n - 1, -1, -1):
        x[i] = z[i]
        x[n - 1] = z[n - 1] / A[n - 1, n - 1]
        if i != n - 1:
            for j in range(n - 1, i, -1):
                x[i] = (x[i] - A[i, j] * x[j])
    return x


def ex41(A, b, x):
    # norma = 0
    # Y = np.zeros(n)
    # for i in range(n):
    #     for j in range(n):
    #         Y[i] += A[i, j] * x[j]
    #     Y[i] -= b[i]
    #     norma += Y[i] ** 2
    # norma = np.sqrt(norma)
    norma = np.linalg.norm(np.dot(A, x, b))
    if norma < eps:
        print("Norma este:", norma)
    else:
        print("Norma nu este mai mica decat epsilon")


def ex42(s, x):
    norma1 = 0
    for i in range(n):
        x[i] = x[i] - s[i]
        norma1 += x[i] ** 2
    norma1 = np.sqrt(norma1)
    norma2 = np.linalg.norm(s)
    norma = norma1 / norma2
    if norma < eps:
        print("Norma este:", norma)
    else:
        print("Norma nu este mai mica decat epsilon")


def ex5(A, q):
    x = np.zeros((n, 1))
    z = np.zeros(((n, 1)))
    for i in range(0, n):
        z[i] = q[i]
        for j in range(0, i):
            z[i] = z[i] - A[i, j] * z[j]
    for i in range(n - 1, -1, -1):
        x[i] = z[i]
        x[n - 1] = z[n - 1] / A[n - 1, n - 1]
        if i != n - 1:
            for j in range(n - 1, i, -1):
                x[i] = (x[i] - A[i, j] * x[j])
    return x


if __name__ == '__main__':
    eps = 10 ** -6
    # Ainit = np.array(([0, 0, 4], [1, 2, 3], [0, 1, 2]))
    Ainit = np.random.randint(0, 2000, size=(10, 10))
    A = Ainit.copy()
    # s = np.array(([3, 2, 1]))
    s = np.random.randint(0, 2000, size=(10, 1))
    n, n = A.shape
    b = ex1(A, s, n)
    q = ex2(A, b, n)
    xQR = ex3QR(Ainit, b)
    xHouseholder = ex3Householder(A, b)
    print("Norma ex 3 este:", np.linalg.norm(xQR - xHouseholder))

    ex41(Ainit, b, xHouseholder)
    ex41(Ainit, b, xQR)
    ex42(s, xHouseholder)
    ex42(s, xQR)
    qinv = np.linalg.inv(q)
    AinvHouseholder = np.empty([n, 1])
    for i in range(0, n):
        Ainv = np.append(AinvHouseholder, ex5(A, qinv[i]), axis=1)
    AinvHouseholder = np.delete(Ainv, np.s_[0:1], axis=1)
    AinvBibl = np.linalg.inv(A)
    norma = np.linalg.norm(AinvHouseholder - AinvBibl)
    print("Norma ex 5:", norma)
