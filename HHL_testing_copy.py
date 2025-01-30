import numpy as np
import scipy as sc
from linear_solvers import NumPyLinearSolver, HHL
import time
from linear_solvers.matrices.tridiagonal_toeplitz import TridiagonalToeplitz
from linear_solvers.matrices.numpy_matrix import NumPyMatrix
from qiskit.quantum_info import Statevector
from qiskit import Aer, transpile
from scipy.sparse import diags
import matplotlib.pyplot as plt
import seaborn as sns


np.set_printoptions(precision=3, suppress=True, linewidth=120)


def get_solution_vector(solution, N, vector):
    """Extracts and normalizes simulated state vector from LinearSolverResult."""
    start = 2**(N-1)
    fin = 2**(N-1) + len(vector)
    solution_vector = Statevector(solution.state).data[start:fin].real
    norm = solution.euclidean_norm
    return norm * solution_vector / np.linalg.norm(solution_vector)


def investigate_error_vs_epsilon(matrix, vector, eps_values):
    """Investigate the error dependence on epsilon."""
    NUM_QUBITS = int(np.log2(matrix.shape[0]))
    numpy_solve = NumPyLinearSolver()
    numpy_solution = numpy_solve.solve(matrix, vector / np.linalg.norm(vector))

    # Lists to store errors
    naive_norms = []
    tridi_norms = []
    aer_norms = []
    cheby_norms = []
    cheby_tri_norms = []

    naive_comp = []
    tridi_comp = []
    aer_comp = []
    cheby_comp = []
    cheby_tri_comp = []

    a = matrix[0][0]  # Just a placeholder extraction
    b = matrix[0][1]  # Adjust according to your matrix definition
    
    print('a:', a)
    print('b:', b) 
    
    # Construct TridiagonalToeplitz matrix with fixed trotter_steps
    tridi_mat = TridiagonalToeplitz(NUM_QUBITS, a, b)
    naive_mat = NumPyMatrix(matrix)
    
    for eps in eps_values:

        # Solve using various methods
        # Naive HHL (no simplification)
        naive_hhl = HHL(epsilon=eps, reciprocal=True)
        naive_solution = naive_hhl.solve(naive_mat, vector)

        # Tridiagonal Toeplitz HHL
        tridi_hhl = HHL(epsilon=eps, reciprocal=True)
        tridi_solution = tridi_hhl.solve(tridi_mat, vector)

        # Aer simulation
        aer_hhl = HHL(epsilon=eps, reciprocal=True, quantum_instance=Aer.get_backend('aer_simulator'))
        aer_solution = aer_hhl.solve(tridi_mat, vector)

        # Chebyshev method
        cheby_hhl = HHL(epsilon=eps, reciprocal=False)
        cheby_tri_solution = cheby_hhl.solve(tridi_mat, vector)
        cheby_solution = cheby_hhl.solve(naive_mat, vector)

        # Compute errors in norms
        naive_norms.append(np.linalg.norm(naive_solution.euclidean_norm - numpy_solution.euclidean_norm)*100)
        tridi_norms.append(np.linalg.norm(tridi_solution.euclidean_norm - numpy_solution.euclidean_norm)*100)
        aer_norms.append(np.linalg.norm(aer_solution.euclidean_norm - numpy_solution.euclidean_norm)*100)
        cheby_tri_norms.append(np.linalg.norm(cheby_tri_solution.euclidean_norm - numpy_solution.euclidean_norm)*100)
        cheby_norms.append(np.linalg.norm(cheby_solution.euclidean_norm - numpy_solution.euclidean_norm)*100)

        # Compute component-wise errors
        naive_comp.append(np.linalg.norm(get_solution_vector(naive_solution, naive_solution.state.num_qubits, vector) - numpy_solution.state)*100)
        tridi_comp.append(np.linalg.norm(get_solution_vector(tridi_solution, tridi_solution.state.num_qubits, vector) - numpy_solution.state)*100)
        aer_comp.append(np.linalg.norm(get_solution_vector(aer_solution, aer_solution.state.num_qubits, vector) - numpy_solution.state)*100)
        cheby_tri_comp.append(np.linalg.norm(get_solution_vector(cheby_tri_solution, cheby_tri_solution.state.num_qubits, vector) - numpy_solution.state)*100)
        cheby_comp.append(np.linalg.norm(get_solution_vector(cheby_solution, cheby_solution.state.num_qubits, vector) - numpy_solution.state)*100)
        
        print(f"Completed epsilon = {eps}")
        print('naive_norms:', naive_norms)
        print('tridi_norms:', tridi_norms)
        print('aer_norms:', aer_norms)
        print('cheby_tri_norms:', cheby_tri_norms)
        print('cheby_norms:', cheby_norms)
        print('')
        print('naive_comp:', naive_comp)
        print('tridi_comp:', tridi_comp)
        print('aer_comp:', aer_comp)
        print('cheby_tri_comp:', cheby_tri_comp)
        print('cheby_comp:', cheby_comp)
        
    # Plot results for Norm error
    eps_values = np.log10(eps_values)
    print(naive_norms)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=eps_values, y=naive_norms, label='Naive HHL', color='b', marker='o', ax=ax1)
    sns.lineplot(x=eps_values, y=aer_norms, label='Aer Tridiagonal', color='k', marker='x', ax=ax1)
    sns.lineplot(x=eps_values, y=tridi_norms, label='Tridiagonal Toeplitz', color='r', marker='s', ax=ax1)
    ax1.set_xlabel('log10 Epsilon')
    ax1.set_ylabel('Error / %')
    ax1.set_title('Error in Euclidean Norm vs Epsilon')
    ax2 = ax1.twinx()
    # If you have cheby_norms for direct Chebyshev (without tridi), you can plot them as well
    sns.lineplot(x=eps_values, y=cheby_tri_norms, label='Chebyshev Tridiagonal', color='c', marker='D', ax=ax2)
    sns.lineplot(x=eps_values, y=cheby_norms, label='Chebyshev', color='g', marker='^', ax=ax2)
    ax2.set_ylabel('Error (Chebyshev) / %')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.savefig('error_in_euclidean_norm_vs_epsilon_N8.png')
    plt.show()

    # Plot results for Component-wise error
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=eps_values, y=naive_comp, label='Naive HHL', color='b', marker='o', ax=ax1)
    sns.lineplot(x=eps_values, y=aer_comp, label='Aer Simulation', color='k', marker='x', ax=ax1)
    sns.lineplot(x=eps_values, y=tridi_comp, label='Tridiagonal Toeplitz', color='r', marker='s', ax=ax1)
    ax1.set_xlabel('log10 Epsilon')
    ax1.set_ylabel('Error / %')
    ax1.set_title('Component wise error in Solution Vector vs Epsilon')
    ax2 = ax1.twinx()
    sns.lineplot(x=eps_values, y=cheby_tri_comp, label='Chebyshev Tridiagonal', color='c', marker='D', ax=ax2)
    sns.lineplot(x=eps_values, y=cheby_comp, label='Chebyshev', color='g', marker='^', ax=ax2)
    ax2.set_ylabel('Error (Chebyshev) / %')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.savefig('component_wise_error_vs_epsilon_N8.png')
    plt.show()



def investigate_error_vs_trotter(matrix, vector, trotter_steps_list, eps=1e-3):
    """Investigate the error dependence on trotter steps, including:
       - Errors in Euclidean norm and component-wise
       - Execution time for each scheme
       - Circuit depth for each scheme
    """
    NUM_QUBITS = int(np.log2(matrix.shape[0]))
    numpy_solve = NumPyLinearSolver()
    numpy_solution = numpy_solve.solve(matrix, vector / np.linalg.norm(vector))

    # Lists to store errors in Euclidean norm
    tridi_norms = []
    aer_norms = []
    cheby_tri_norms = []
    naive_norms = []
    cheby_norms = []

    # Lists to store component-wise errors
    tridi_comp = []
    aer_comp = []
    cheby_tri_comp = []
    naive_comp = []
    cheby_comp = []

    # Lists to store execution times
    naive_times = []
    tridi_times = []
    aer_times = []
    cheby_times = []
    cheby_tri_times = []

    # Lists to store circuit depths
    naive_depths = []
    tridi_depths = []
    aer_depths = []
    cheby_depths = []
    cheby_tri_depths = []

    a = matrix[0][0]
    b = matrix[0][1]
    print('a:', a)
    print('b:', b)

    # Set style for plots
    #sns.set_style("whitegrid")
    naive_mat = NumPyMatrix(matrix)
    
    for steps in trotter_steps_list:
        tridi_mat = TridiagonalToeplitz(NUM_QUBITS, a, b, trotter_steps=steps)
        print("\nCurrently running with trotter_steps =", tridi_mat.trotter_steps, "\n")

        # Naive HHL (no simplification)
        start_time = time.time()
        naive_hhl = HHL(epsilon=eps, reciprocal=True)
        naive_solution = naive_hhl.solve(naive_mat, vector)
        naive_times.append(time.time() - start_time)

        # Tridiagonal Toeplitz HHL
        start_time = time.time()
        tridi_hhl = HHL(epsilon=eps, reciprocal=True)
        tridi_solution = tridi_hhl.solve(tridi_mat, vector)
        tridi_times.append(time.time() - start_time)

        # Aer simulation
        start_time = time.time()
        aer_hhl = HHL(epsilon=eps, reciprocal=True, quantum_instance=Aer.get_backend('aer_simulator'))
        aer_solution = aer_hhl.solve(tridi_mat, vector)
        aer_times.append(time.time() - start_time)

        # Chebyshev method (non-tridi)
        start_time = time.time()
        cheby_hhl = HHL(epsilon=eps, reciprocal=False)
        cheby_solution = cheby_hhl.solve(naive_mat, vector)
        cheby_times.append(time.time() - start_time)

        # Chebyshev method (tridi)
        start_time = time.time()
        cheby_tri_solution = cheby_hhl.solve(tridi_mat, vector)
        cheby_tri_times.append(time.time() - start_time)

        # Compute errors in Euclidean norms
        naive_norms.append(np.linalg.norm(naive_solution.euclidean_norm - numpy_solution.euclidean_norm)*100)
        tridi_norms.append(np.linalg.norm(tridi_solution.euclidean_norm - numpy_solution.euclidean_norm)*100)
        aer_norms.append(np.linalg.norm(aer_solution.euclidean_norm - numpy_solution.euclidean_norm)*100)
        cheby_norms.append(np.linalg.norm(cheby_solution.euclidean_norm - numpy_solution.euclidean_norm)*100)
        cheby_tri_norms.append(np.linalg.norm(cheby_tri_solution.euclidean_norm - numpy_solution.euclidean_norm)*100)

        # Compute component-wise errors
        naive_comp.append(np.linalg.norm(get_solution_vector(naive_solution, naive_solution.state.num_qubits, vector) - numpy_solution.state)*100)
        tridi_comp.append(np.linalg.norm(get_solution_vector(tridi_solution, tridi_solution.state.num_qubits, vector) - numpy_solution.state)*100)
        aer_comp.append(np.linalg.norm(get_solution_vector(aer_solution, aer_solution.state.num_qubits, vector) - numpy_solution.state)*100)
        cheby_comp.append(np.linalg.norm(get_solution_vector(cheby_solution, cheby_solution.state.num_qubits, vector) - numpy_solution.state)*100)
        cheby_tri_comp.append(np.linalg.norm(get_solution_vector(cheby_tri_solution, cheby_tri_solution.state.num_qubits, vector) - numpy_solution.state)*100)

        # Compute circuit depths
        naive_qc = transpile(naive_solution.state, basis_gates=['id', 'rz', 'sx', 'x', 'cx'])
        tridi_qc = transpile(tridi_solution.state, basis_gates=['id', 'rz', 'sx', 'x', 'cx'])
        aer_qc = transpile(aer_solution.state, basis_gates=['id', 'rz', 'sx', 'x', 'cx'])
        cheby_qc = transpile(cheby_solution.state, basis_gates=['id', 'rz', 'sx', 'x', 'cx'])
        cheby_tri_qc = transpile(cheby_tri_solution.state, basis_gates=['id', 'rz', 'sx', 'x', 'cx'])

        naive_depths.append(naive_qc.depth())
        tridi_depths.append(tridi_qc.depth())
        aer_depths.append(aer_qc.depth())
        cheby_depths.append(cheby_qc.depth())
        cheby_tri_depths.append(cheby_tri_qc.depth())
        
        print('naive_norms:', naive_norms)
        print('tridi_norms:', tridi_norms)
        print('aer_norms:', aer_norms)
        print('cheby_tri_norms:', cheby_tri_norms)
        print('cheby_norms:', cheby_norms)
        print('')
        print('naive_comp:', naive_comp)
        print('tridi_comp:', tridi_comp)
        print('aer_comp:', aer_comp)
        print('cheby_tri_comp:', cheby_tri_comp)
        print('cheby_comp:', cheby_comp)


    # Plot: Error in Euclidean Norm vs Trotter Steps
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=trotter_steps_list, y=naive_norms, label='Naive HHL', color='b', marker='o', ax=ax1)
    sns.lineplot(x=trotter_steps_list, y=aer_norms, label='Aer Tridiagonal', color='k', marker='x', ax=ax1)
    sns.lineplot(x=trotter_steps_list, y=tridi_norms, label='Tridiagonal Toeplitz', color='r', marker='s', ax=ax1)
    ax1.set_xlabel('Trotter Steps')
    ax1.set_ylabel('Error / %')
    ax2 = ax1.twinx()
    sns.lineplot(x=trotter_steps_list, y=cheby_norms, label='Chebyshev', color='g', marker='^', ax=ax2)
    sns.lineplot(x=trotter_steps_list, y=cheby_tri_norms, label='Chebyshev Tridiagonal', color='c', marker='D', ax=ax2)
    ax2.set_ylabel('Error (Chebyshev) / %')
    ax1.set_title('Error in Euclidean Norm vs Trotter Steps')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.savefig('error_in_euclidean_norm_vs_trotter_steps_N8.png')
    plt.show()

    # Plot: Component-wise Error vs Trotter Steps
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=trotter_steps_list, y=naive_comp, label='Naive HHL', color='b', marker='o', ax=ax1)
    sns.lineplot(x=trotter_steps_list, y=aer_comp, label='Aer Simulation', color='k', marker='x', ax=ax1)
    sns.lineplot(x=trotter_steps_list, y=tridi_comp, label='Tridiagonal Toeplitz', color='r', marker='s', ax=ax1)
    ax1.set_xlabel('Trotter Steps')
    ax1.set_ylabel('Error / %')
    ax1.set_title('Component-wise Error vs Trotter Steps')
    ax2 = ax1.twinx()
    sns.lineplot(x=trotter_steps_list, y=cheby_comp, label='Chebyshev', color='g', marker='^', ax=ax2)
    sns.lineplot(x=trotter_steps_list, y=cheby_tri_comp, label='Chebyshev Tridiagonal', color='c', marker='D', ax=ax2)
    ax2.set_ylabel('Error (Chebyshev) / %')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.savefig('component_wise_error_vs_trotter_steps_N8.png')
    plt.show()

    # Plot: Execution time vs Trotter Steps
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=trotter_steps_list, y=naive_times, label='Naive HHL', color='b', marker='o', ax=ax)
    sns.lineplot(x=trotter_steps_list, y=aer_times, label='Aer Simulation', color='k', marker='x', ax=ax)
    sns.lineplot(x=trotter_steps_list, y=tridi_times, label='Tridiagonal Toeplitz', color='r', marker='s', ax=ax)
    sns.lineplot(x=trotter_steps_list, y=cheby_times, label='Chebyshev', color='g', marker='^', ax=ax)
    sns.lineplot(x=trotter_steps_list, y=cheby_tri_times, label='Chebyshev Tridiagonal', color='c', marker='D', ax=ax)
    ax.set_xlabel('Trotter Steps')
    ax.set_ylabel('Execution Time (s)')
    ax.set_title('Execution Time vs Trotter Steps')
    ax.legend(loc='upper left')
    plt.savefig('execution_time_vs_trotter_steps_N8.png')
    plt.show()

    # Plot: Circuit Depth vs Trotter Steps
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=trotter_steps_list, y=naive_depths, label='Naive HHL', color='b', marker='o', ax=ax)
    sns.lineplot(x=trotter_steps_list, y=aer_depths, label='Aer Simulation', color='k', marker='x', ax=ax)
    sns.lineplot(x=trotter_steps_list, y=tridi_depths, label='Tridiagonal Toeplitz', color='r', marker='s', ax=ax)
    sns.lineplot(x=trotter_steps_list, y=cheby_depths, label='Chebyshev', color='g', marker='^', ax=ax)
    sns.lineplot(x=trotter_steps_list, y=cheby_tri_depths, label='Chebyshev Tridiagonal', color='c', marker='D', ax=ax)
    ax.set_xlabel('Trotter Steps')
    ax.set_ylabel('Circuit Depth')
    ax.set_title('Circuit Depth vs Trotter Steps')
    ax.legend(loc='upper left')
    plt.savefig('circuit_depth_vs_trotter_steps_N8.png')
    plt.show()
    
    
def investigate_error_vs_condition_number(base_matrix, vector, condition_numbers, eps=1e-3):
    """Investigate the error dependence on the condition number of the matrix."""
    numpy_solve = NumPyLinearSolver()
    NUM_QUBITS = int(np.log2(matrix.shape[0]))
    tri_errors = []
    nav_errors = []
    cheb_nav_errors = []
    cheb_tri_errors = []

    for cond_num in condition_numbers:
        # Adjust the matrix to achieve the desired condition number
        U, S, Vt = np.linalg.svd(base_matrix)  # Singular Value Decomposition
        S_new = np.linspace(cond_num, 1, len(S))  # New singular values
        modified_matrix = U @ np.diag(S_new) @ Vt

        # Compute the reference solution using NumPy
        numpy_solution = numpy_solve.solve(modified_matrix, vector / np.linalg.norm(vector))
        
        # naive 
        naive_mat = NumPyMatrix(matrix)
        naive_hhl = HHL(epsilon=eps, reciprocal=True)
        naive_solution = naive_hhl.solve(naive_mat, vector)

        # Solve using HHL
        a = modified_matrix[0][0] # main diag
        b = modified_matrix[0][1] # Off diag
        tridi_mat = TridiagonalToeplitz(NUM_QUBITS, a, b)
        hhl_solver = HHL(epsilon=eps, reciprocal=True)
        hhl_solution = hhl_solver.solve(tridi_mat, vector)
        
        # Solve with Cheby
        cheby_hhl = HHL(epsilon=eps, reciprocal=False)
        cheby_solution_nav = cheby_hhl.solve(naive_mat, vector)
        cheby_solution_tri = cheby_hhl.solve(tridi_mat, vector)

        # Compute the error in Euclidean norm
        tri_errors.append(np.linalg.norm(hhl_solution.euclidean_norm - numpy_solution.euclidean_norm) * 100)
        nav_errors.append(np.linalg.norm(naive_solution.euclidean_norm - numpy_solution.euclidean_norm) * 100)
        cheb_tri_errors.append(np.linalg.norm(cheby_solution_tri.euclidean_norm - numpy_solution.euclidean_norm) * 100)
        cheb_nav_errors.append(np.linalg.norm(cheby_solution_nav.euclidean_norm - numpy_solution.euclidean_norm) * 100)
        

        print(f"Condition number: {cond_num}, Tridiag Error: {tri_errors}%")
        print(f"Condition number: {cond_num}, Naive Error: {nav_errors}%")
        print(f"Condition number: {cond_num}, Chebyshev Tridiag Error: {cheb_tri_errors}%")
        print(f"Condition number: {cond_num}, Chebyshev Naive Error: {cheb_nav_errors}%")
        
        print('')
        
        print('matrix eigs \n', np.linalg.eigvals(modified_matrix))

    # Plot error vs condition number
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot errors for Tridiagonal Toeplitz and Naive HHL
    sns.lineplot(x=condition_numbers, y=tri_errors, color='r', marker='o', label='Tridiagonal Toeplitz', ax=ax1)
    sns.lineplot(x=condition_numbers, y=nav_errors, color='b', marker='s', label='Naive HHL', ax=ax1)

    ax1.set_xlabel("Condition Number")
    ax1.set_ylabel("Error (%)")
    ax1.set_title("Error vs Condition Number")

    # Create a twin y-axis for Chebyshev errors
    sns.lineplot(x=condition_numbers, y=cheb_tri_errors, color='g', marker='^', label='Chebyshev Tridiagonal', ax=ax1)
    sns.lineplot(x=condition_numbers, y=cheb_nav_errors, color='c', marker='D', label='Chebyshev', ax=ax1)

    # Add legends for both axes
    ax1.legend(loc='upper right')

    # Add grid
    ax1.grid(True)
    
    #plt.savefig('error_vs_condition_number_N8.png')

    # Show the plot
    plt.show()


# to make the code use the acutal HHL keep it in this location


# Example usage:
if __name__ == "__main__":
    NUM_QUBITS = 2
    MATRIX_SIZE = 2 ** NUM_QUBITS

    a = -2
    b = 1
    matrix = diags([b, a, b], [-1, 0, 1], shape=(MATRIX_SIZE, MATRIX_SIZE)).toarray()
    #vector = np.array([1] + [0]*(MATRIX_SIZE - 1))
    vector = np.sin(np.linspace(0, np.pi, MATRIX_SIZE))
    
    print('matrix\n' , matrix)
    print('vector\n', vector)
    print('matrix condition number:', np.linalg.cond(matrix))
    print('matrix eigs \n', np.linalg.eigvals(matrix))
    
    # Create matrix circuit quantum instance
    tridi_mat = TridiagonalToeplitz(NUM_QUBITS, a, b)
    # Tridiagonal Toeplitz HHL
    tridi_hhl = HHL()
    tridi_solution = tridi_hhl.solve(tridi_mat, vector)
    
    tridi_sol_vec = get_solution_vector(tridi_solution, tridi_solution.state.num_qubits, vector)
    print(tridi_sol_vec)

    # Investigate error vs epsilon
    eps_values = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    #investigate_error_vs_epsilon(matrix, vector, eps_values )
    #print('Investigation of error vs epsilon completed.')
    
    # Investigate error vs trotter steps
    Tsteps = [2, 3, 4, 5, 6, 7, 8]
    #investigate_error_vs_trotter(matrix, vector, Tsteps, eps=1e-3)
    
    # Investigate error vs condition number
    condition_numbers = [1, 1.5, 2, 5, 7, 9, 11, 13]
    #investigate_error_vs_condition_number(matrix, vector, condition_numbers)
