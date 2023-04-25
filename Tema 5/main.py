import numpy as np
import scipy.stats as stats
import scipy.sparse as sparse
import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigs


def sprandsym(n, density):
    np.random.seed(42)
    rvs = stats.norm(loc=3, scale=1).rvs
    X = sparse.random(n, n, density=density, data_rvs=rvs)
    upper_X = sparse.triu(X)
    # facem matricea rara simetrica
    result = upper_X + upper_X.T - sparse.diags(X.diagonal())
    return result

M = sprandsym(5000, 0.5)


def ex1(file):
    index = 0
    data = []
    row = []
    col = []
    for line in open(file):
        if index == 0:
            n = int(line.strip())
            index += 1
        else:
            val, i, j = line.split(",")
            val = float(val.strip())
            i = int(i.strip())
            j = int(j.strip())
            data.append(val)
            row.append(i)
            col.append(j)
    m = sp.coo_matrix((data, (row, col)))
    return m, n


def verif_sym(coo_matrix):
    coo_matrix_transpose = coo_matrix.transpose()
    is_symmetric = (coo_matrix.data == coo_matrix_transpose.data).all() and \
                   (coo_matrix.row == coo_matrix_transpose.row).all() and \
                   (coo_matrix.col == coo_matrix_transpose.col).all()

    if is_symmetric:
        return True
    return False


def met_puterii(m, k_max=1000000):
    n = m.shape[0]
    eps = 10 ** -9
    x = np.random.rand(n)
    x = x/np.linalg.norm(x)

    val = 0.0
    vec = np.zeros(0)
    for i in range(0, k_max):
        x_new = m @ x
        val_new = np.linalg.norm(x_new)
        x_new = x_new / val_new

        if np.abs(val_new - val) < eps:
            val = val_new
            vec = x_new
            break
        val = val_new
        x = x_new

    return val, vec


def ex_3():
    eps = 10 ** -9
    contor = 0
    p = 5
    n = 3
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [3, 3, 3], [4, 6, 8]])
    b = np.array([4, 5, 6, 7, 8])
    U, s, V = np.linalg.svd(A)
    print("Valorile sigulare ale matricei A sunt:", s)
    # print("rang(A) = ", np.linalg.matrix_rank(A))
    for i in range(0, len(s)):
        if s[i] > eps:
            contor += 1
    print("rang(A)= ", contor)
    # print("numarul de conditionare al matricei A:",np.linalg.cond(A) )
    print("numarul de conditionare al matricei A", max(s)/min(s))
    print("Pseudoinversa:", np.linalg.pinv(A))
    # for i in range(0, len(s)):
    #     s[i] = 1/s[i]
    # s_new = np.diag(s)
    # for i in range(len(s_new),p):
    #     s_new = np.vstack((s_new,np.zeros(n)))
    # print(s_new)
    # x = V.T @ np.linalg.inv(s_new)
    # print("pseudo", x @ U.T)

    # Calculul soluției sistemului Ax = b
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    # Calculul produsului A * x
    Ax = np.dot(A, x)

    # Calculul normei ||b - Ax||2
    norma = np.linalg.norm(b - Ax, ord=2)

    # Afișarea soluției și normei
    print("Solutia sistemului Ax = b:\n", x)
    print("Norma ||b - Ax||2:\n", norma)


if __name__ == "__main__":
    m1, n1 = ex1("m2.txt")
    # print(m1)
    # print(M) # matricea generata de noi
    # print(verif_sym(m1))
    vall, vecc = met_puterii(m1)
    print("Valoarea proprie aprox", vall)
    print("Vectorul propriu asociat", vecc)

    ex_3()






