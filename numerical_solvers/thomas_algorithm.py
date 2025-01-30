#!/usr/bin/env python3
"""
Example: Thomas Algorithm in Python

This script defines a class `numerical_solvers` containing a method
that solves a tridiagonal linear system A x = b using the Thomas algorithm.
"""

import numpy as np

class numerical_solvers:
    """
    A collection of numerical solver methods for linear systems.
    """

    @staticmethod
    def solve_tridiagonal_thomas(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve the linear system A x = b, where A is an n x n tridiagonal matrix,
        using the Thomas (tridiagonal) algorithm.

        Parameters
        ----------
        A : np.ndarray
            Tridiagonal coefficient matrix of shape (n, n).
            Assumes A[i,i], A[i,i-1], and A[i,i+1] may be non-zero
            and all other entries are zero.
        b : np.ndarray
            Right-hand side vector of shape (n,).

        Returns
        -------
        x : np.ndarray
            The solution vector of shape (n,).

        Notes
        -----
        This implementation modifies A and b internally for efficiency;
        if you need to preserve them, pass copies instead (e.g., A.copy(), b.copy()).
        """
        # Number of equations
        n = A.shape[0]

        # Extract the diagonals (sub, main, and super) from A
        # a[i] = A[i, i-1], b_diag[i] = A[i, i], c[i] = A[i, i+1]
        # We'll rename to avoid confusion with input vector b:
        a_sub = np.zeros(n)      # sub-diagonal (below main)
        b_diag = np.zeros(n)     # main diagonal
        c_sup = np.zeros(n)      # super-diagonal (above main)
        d = b.astype(float).copy()  # RHS (make a float copy to avoid integer division issues)

        for i in range(n):
            b_diag[i] = A[i, i]
            if i > 0:
                a_sub[i] = A[i, i - 1]
            if i < n - 1:
                c_sup[i] = A[i, i + 1]

        # Forward elimination
        for i in range(1, n):
            w = a_sub[i] / b_diag[i - 1]
            b_diag[i] = b_diag[i] - w * c_sup[i - 1]
            d[i] = d[i] - w * d[i - 1]

        # Back substitution
        x = np.zeros(n, dtype=float)
        x[n - 1] = d[n - 1] / b_diag[n - 1]
        for i in range(n - 2, -1, -1):
            x[i] = (d[i] - c_sup[i] * x[i + 1]) / b_diag[i]

        return x


# --------------------------------------------------------------------
# Example usage (uncomment to test):
# --------------------------------------------------------------------
# if __name__ == "__main__":
#     # Example: A 4x4 tridiagonal system
#     A_mat = np.array([
#         [ 2.0, -1.0,  0.0,  0.0],
#         [-1.0,  2.0, -1.0,  0.0],
#         [ 0.0, -1.0,  2.0, -1.0],
#         [ 0.0,  0.0, -1.0,  2.0]
#     ])
#     b_vec = np.array([1.0, 0.0, 0.0, 1.0])
#
#     solver = numerical_solvers()
#     x_solution = solver.solve_tridiagonal_thomas(A_mat, b_vec)
#     print("Solution x =", x_solution)
#
#     # You can verify the result by checking np.allclose(A_mat @ x_solution, b_vec)
