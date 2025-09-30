import numpy as np
import matplotlib.pyplot as plt

# Forward Substitution
def forward_substitution(L, b):
    """ Solve L * y = b using forward substitution where L is a lower triangular matrix. """
    y = np.zeros_like(b)
    for i in range(len(b)):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y

# Back Substitution
def back_substitution(U, y):
    """ Solve U * x = y using back substitution where U is an upper triangular matrix. """
    x = np.zeros_like(y)
    for i in range(len(y)-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
        print(U[i, i], y[i], np.dot(U[i, i+1:], x[i+1:]))
    return x

# Cholesky Least Squares Implementation
def cholesky_least_squares(A, b):
    # Compute A.T @ A
    ATA = A.T @ A
    # Compute A.T @ b
    ATb = A.T @ b
    # Cholesky decomposition: ATA = R.T @ R
    R = np.linalg.cholesky(ATA)
    # Solve R @ y = A.T @ b using forward substitution (because R is lower triangular)
    y = forward_substitution(R, ATb)
    # Solve R.T @ x = y using back substitution (because R.T is upper triangular)
    x = back_substitution(R.T, y)
    return x

# Second dataset
n = 30
x_data = np.linspace(-2, 2, n)
eps = 1
np.random.seed(1)
r = np.random.rand(n) * eps


y_data1 = 4*x_data**5 - 5*x_data**4 - 20*x_data**3 + 10*x_data**2 + 40*x_data + 10 + r

# Construct the Vandermonde matrix A ( m = 3)
m = 3
A = np.vander(x_data, m, increasing=True)

# Solve least squares for the first dataset
coefficients1 = cholesky_least_squares(A, y_data1)

# Compute the fitted polynomial 
fit1 = A @ coefficients1

# Plot first dataset with m=8
plt.figure()
plt.scatter(x_data, y_data1, label='Data 2', color='blue')
plt.plot(x_data, fit1, label='Fit 1 (m=3)', color='red')
plt.title('Polynomial Fit for Data 2 (m=3)')
plt.legend()

plt.show()