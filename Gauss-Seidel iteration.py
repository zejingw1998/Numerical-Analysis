import numpy as np

def GaussSeidel(A,b,tol=1e-10, max_iter=1000):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    x = np.zeros(n, dtype=float)

    for _ in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i],     x[:i])
            s2 = np.dot(A[i, i+1:], x_old[i+1:])
            if A[i, i] == 0:
                raise ZeroDivisionError("A[i,i]=0")
            x[i] = (b[i] - s1 - s2) / A[i, i]

        # Convergence
        if np.linalg.norm(x - x_old, np.inf) < tol:
            break
    return x
    
    
A = np.array([[2, 0, 0],
              [1, 3, 0],
              [-1, 2, 5]], dtype=float)
b = np.array([4, 7, 2], dtype=float)

x = GaussSeidel(A, b)
print("x =", x)