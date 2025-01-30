import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import Initialize
from linear_solvers.matrices import NumPyMatrix
import scipy as sp
import matplotlib.pyplot as plt
from typing import List



NUM_QUBITS = 1
MATRIX_SIZE = 2 ** NUM_QUBITS

a = 1
b = -1/6
matrix = sp.sparse.diags([b, a, b], [-1, 0, 1], shape=(MATRIX_SIZE, MATRIX_SIZE)).toarray()
vector = np.array([1] + [0]*(MATRIX_SIZE - 1))

kappa = np.linalg.cond(matrix)
eps = 1e-2 

print(' \n Matrix: \n', matrix)
print(' \n Vector: \n', vector)



J = (kappa/eps)*(np.log(kappa/eps))
J = int(np.ceil(J))
K = kappa * np.log(kappa/eps)
K = int(np.ceil(K))

delta_y = eps/(np.sqrt(np.log(kappa/eps)))
delta_z = 1/(kappa*np.sqrt(np.log(kappa/eps)))

y_vals = np.array([delta_y*i for i in range(J)])
z_vals = np.array([delta_z*i for i in range(-K, K+1)])

print('\n Params. : \n  J: ', J, '\n K: ', K, '\n delta_y: ', delta_y, '\n delta_z: ', delta_z)

matrix = NumPyMatrix(matrix)
print('\n NumPyMatrix: \n', matrix.power(1))