import numpy as np

def forward(A, b):
    n = A.shape[0]
    x = np.zeros_like(b, dtype=float)
    
    for i in range(n): 
        x[i] = (b[i] - A[i, :i] @ x[:i]) / A[i, i]
    return x



A = np.array([[2, 0, 0],
              [1, 3, 0],
              [-1, 2, 5]], dtype=float)
b = np.array([4, 7, 2], dtype=float)

x = forward(A, b)
print("x =", x)
