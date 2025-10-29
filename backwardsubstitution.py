import numpy as np

def backsolve(A, b):
    """
    Solve the upper triangular system A x = b
    using backward substitution.
    """
    n = A.shape[0]
    x = np.zeros_like(b, dtype=float)
    
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - A[i, i+1:] @ x[i+1:]) / A[i, i]

    
    return x

A = np.array([[2, 1, -1],
              [0, 3, 2],
              [0, 0, 5]])
b = np.array([3, 7, 10])

x = backsolve(A, b)
print("x =", x)



