import numpy as np

A1 = np.eye(5)
A2 = 2 * A1
A3 = 3 * A1
A4 = 4 * A1

# Create a 3D array from the matrices
A = np.array([A1, A2, A3, A4])
lambdas = np.array([1, 0.5, 1/3])

# Reshape lambdas for broadcasting
lambdas = lambdas[:, np.newaxis, np.newaxis]

# Multiply the stacked matrices by the lambdas

N = len(lambdas)
result = np.sum(A[0:N,:,:] * lambdas,axis=0)

print(result)