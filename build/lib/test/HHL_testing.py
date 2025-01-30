import numpy as np
# pylint: disable=line-too-long
from linear_solvers import NumPyLinearSolver, HHL

np.set_printoptions(precision=3, suppress=True, linewidth=120)  # Suppress scientific notation, limit decimals, and set max width.

matrix = np.array([[1, -1/3], [-1/3, 1]])
vector = np.array([1, 0])
# solving the system with no simplification
naive_hhl_solution = HHL().solve(matrix, vector)

print('')
print('Matrix:')
print(matrix)
print('')

print('')
print('Vector:')
print(vector)
print('')


classical_solution = NumPyLinearSolver().solve(matrix,
                                               vector/np.linalg.norm(vector))
# The classical solutions has objects:
# state: the solution vector or the circuit which prepares the solution vector
# euclidean_norm: the euclidean norm of the solution vector
# observable: the list of calculated observables
# circuit_results: the observable results of the circuit

print('')
print('Solution:')
print(classical_solution.state)
print('')


from linear_solvers.matrices.tridiagonal_toeplitz import TridiagonalToeplitz
# TridiagonalToeplitz(num_qubits, a, b) where a is main diagonal and b is subdiagonals
tridi_matrix = TridiagonalToeplitz(1, 1, -1 / 3)
print(tridi_matrix)
tridi_solution = HHL().solve(tridi_matrix, vector)

import numpy as np
from qiskit import QuantumCircuit
from linear_solvers.matrices.numpy_matrix import NumPyMatrix

matrix = NumPyMatrix(np.array([[1 / 2, 1 / 6, 0, 0], [1 / 6, 1 / 2, 1 / 6, 0],
                    [0, 1 / 6, 1 / 2, 1 / 6], [0, 0, 1 / 6, 1 / 2]]))
power = 2

num_qubits = matrix.num_state_qubits
# Controlled power (as used within QPE)
pow_circ = matrix.power(power).control()
circ_qubits = pow_circ.num_qubits
qc = QuantumCircuit(circ_qubits)
qc.append(matrix.power(power).control(), list(range(circ_qubits)))

print('')
print(qc.draw())

print('')
print('naive state:')
print(naive_hhl_solution.state)
print('tridiagonal state:')
print(tridi_solution.state)
print('')

# The Euclidean norm of the solution vector 
print('')
print('classical Euclidean norm:', classical_solution.euclidean_norm)
print('naive Euclidean norm:', naive_hhl_solution.euclidean_norm)
print('tridiagonal Euclidean norm:', tridi_solution.euclidean_norm)
print('')


from qiskit.quantum_info import Statevector

# Now extract the actual values of the solution vectors
naive_sv = Statevector(naive_hhl_solution.state).data
tridi_sv = Statevector(tridi_solution.state).data

# Extract vector components; 10000(bin) == 16 & 10001(bin) == 17
naive_full_vector = np.array([naive_sv[16], naive_sv[17]])
tridi_full_vector = np.array([tridi_sv[16], tridi_sv[17]])

# Complex part is near 0, so we can ignore it
print('')
print('naive raw solution vector:', naive_full_vector)
print('tridi raw solution vector:', tridi_full_vector)
print('')

def get_solution_vector(solution):
    """Extracts and normalizes simulated state vector
    from LinearSolverResult."""
    solution_vector = Statevector(solution.state).data[16:18].real
    norm = solution.euclidean_norm
    return norm * solution_vector / np.linalg.norm(solution_vector)
print('')
print('full naive solution vector:', get_solution_vector(naive_hhl_solution))
print('full tridi solution vector:', get_solution_vector(tridi_solution))
print('classical state:', classical_solution.state)
print('')

from scipy.sparse import diags

NUM_QUBITS = 2
MATRIX_SIZE = 2 ** NUM_QUBITS
# entries of the tridiagonal Toeplitz symmetric matrix
# pylint: disable=invalid-name
a = 1
b = -1/3

matrix = diags([b, a, b],
               [-1, 0, 1],
               shape=(MATRIX_SIZE, MATRIX_SIZE)).toarray()

vector = np.array([1] + [0]*(MATRIX_SIZE - 1))
# run the algorithms
classical_solution = NumPyLinearSolver(
                        ).solve(matrix, vector / np.linalg.norm(vector))
naive_hhl_solution = HHL().solve(matrix, vector)
tridi_matrix = TridiagonalToeplitz(NUM_QUBITS, a, b)
tridi_solution = HHL().solve(tridi_matrix, vector)

print('')
print('classical euclidean norm:', classical_solution.euclidean_norm)
print('naive euclidean norm:', naive_hhl_solution.euclidean_norm)
print('tridiagonal euclidean norm:', tridi_solution.euclidean_norm)
print('')

from qiskit import transpile

MAX_QUBITS = 4
a = 1
b = -1/3

i = 1
# calculate the circuit depths for different number of qubits to compare the use
# of resources (WARNING: This will take a while to execute)
naive_depths = []
tridi_depths = []
for n_qubits in range(1, MAX_QUBITS+1):
    matrix = diags([b, a, b],
                   [-1, 0, 1],
                   shape=(2**n_qubits, 2**n_qubits)).toarray()
    vector = np.array([1] + [0]*(2**n_qubits -1))

    naive_hhl_solution = HHL().solve(matrix, vector)
    tridi_matrix = TridiagonalToeplitz(n_qubits, a, b)
    tridi_solution = HHL().solve(tridi_matrix, vector)

    naive_qc = transpile(naive_hhl_solution.state,
                         basis_gates=['id', 'rz', 'sx', 'x', 'cx'])
    tridi_qc = transpile(tridi_solution.state,
                         basis_gates=['id', 'rz', 'sx', 'x', 'cx'])

    naive_depths.append(naive_qc.depth())
    tridi_depths.append(tridi_qc.depth())
    i +=1

print('')
sizes = [f"{2**n_qubits}×{2**n_qubits}"
         for n_qubits in range(1, MAX_QUBITS+1)]
columns = ['size of the system',
           'quantum_solution depth',
           'tridi_solution depth']
data = np.array([sizes, naive_depths, tridi_depths])
ROW_FORMAT ="{:>23}" * (len(columns) + 2)
for team, row in zip(columns, data):
    print(ROW_FORMAT.format(team, *row))
print('')


print('excess:',
      [naive_depths[i] - tridi_depths[i] for i in range(0, len(naive_depths))])

from linear_solvers.observables import AbsoluteAverage, MatrixFunctional

NUM_QUBITS = 1
MATRIX_SIZE = 2 ** NUM_QUBITS
# entries of the tridiagonal Toeplitz symmetric matrix
a = 1
b = -1/3

matrix = diags([b, a, b],
               [-1, 0, 1],
               shape=(MATRIX_SIZE, MATRIX_SIZE)).toarray()
vector = np.array([1] + [0]*(MATRIX_SIZE - 1))
tridi_matrix = TridiagonalToeplitz(1, a, b)

average_solution = HHL().solve(tridi_matrix,
                               vector,
                               AbsoluteAverage())
classical_average = NumPyLinearSolver(
                        ).solve(matrix,
                                vector / np.linalg.norm(vector),
                                AbsoluteAverage())

print('')
print('quantum average:', average_solution.observable)
print('classical average:', classical_average.observable)
print('quantum circuit results:', average_solution.circuit_results)
print('')

observable = MatrixFunctional(1, 1 / 2)

functional_solution = HHL().solve(tridi_matrix, vector, observable)
classical_functional = NumPyLinearSolver(
                          ).solve(matrix,
                                  vector / np.linalg.norm(vector),
                                  observable)

print('')
print('quantum functional:', functional_solution.observable)
print('classical functional:', classical_functional.observable)
print('quantum circuit results:', functional_solution.circuit_results)
print('')

from qiskit import Aer

backend = Aer.get_backend('aer_simulator')
hhl = HHL(1e-3, quantum_instance=backend)

accurate_solution = hhl.solve(matrix, vector)
classical_solution = NumPyLinearSolver(
                    ).solve(matrix,
                            vector / np.linalg.norm(vector))

print('')
print(accurate_solution.euclidean_norm)
print(classical_solution.euclidean_norm)
print('')