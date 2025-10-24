import numpy as np

def my_LUfactorization(A):
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros_like(A)

    for k in range(n):
       
        for j in range(k, n):
            U[k, j] = A[k, j] - np.sum(L[k, :k] * U[:k, j])
        
        for i in range(k + 1, n):
            L[i, k] = (A[i, k] - np.sum(L[i, :k] * U[:k, k])) / U[k, k]
    return L, U

n = 4
A = np.array([[2., 1., 1., 0.],
              [4., -6., 0., 2.],
              [-2., 7., 2., 1.],
              [1., 0., 3., 5.]])

L, U = my_LUfactorization(A)
print("L=\n", L)
print("U=\n", U)
print("LU≈A? ", np.allclose(L @ U, A))
