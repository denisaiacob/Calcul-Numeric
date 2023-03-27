import numpy as np

eps = 10 ** -6


def ex1(file):
    index = 0
    ok = True
    for line in open(file):
        if index == 0:
            n = int(line.strip())
            m = list()
            for k in range(0, n):
                m.append(list())
            index += 1
        else:
            val, i, j = line.split(",")
            val = float(val.strip())
            i = int(i.strip())
            j = int(j.strip())
            if val != 0:
                m[i].append((val, j))
            if i == j and abs(val) < eps:
                ok = False
    # print(ok)
    return m, n


def ex1b(file):
    index = 0
    ok = True
    for line in open(file):
        if index == 0:
            b = list()
            index += 1
        else:
            b.append(float(line.strip()))
    return b


def ex2(a, b, n):
    index = 0
    x = list()
    for i in range(0, n):
        x.append(0)
    for i in range(0, n):
        sum = 0
        for j in range(0, len(a[i])):
            if i != a[i][j][1]:
                sum = sum + a[i][j][0] * x[a[i][j][1]]
            else:
                diag = a[i][j][0]
        x[i] = (b[i] - sum) / diag
        index += 1
    return x


def ex3(a, b, x, n):
    p = np.zeros(n)
    for i in range(0, n):
        suma = 0
        for j in range(0, len(a[i])):
            suma += round(x[i]) * a[i][j][0]
        p[i] = suma

    return np.linalg.norm(p - b)


if __name__ == "__main__":
    a1, n1 = ex1("a1.txt")
    # print(a1)
    # a2, n2 = ex1("a2.txt")
    # a3, n3 = ex1("a3.txt")
    # a4, n4 = ex1("a4.txt")
    # a5, n5 = ex1("a5.txt")
    b1 = ex1b("b1.txt")
    # b2 = ex1b("b2.txt")
    # b3 = ex1b("b3.txt")
    # b4 = ex1b("b4.txt")
    # b5 = ex1b("b5.txt")
    x1 = ex2(a1, b1, n1)
    print(ex3(a1, b1, x1, n1))