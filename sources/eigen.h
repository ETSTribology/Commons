// eigen.h
#ifndef _EIGEN_H
#define _EIGEN_H

#include <stdio.h>
#include <string.h>
#include <math.h>

// Define epsilon constants for floating-point comparisons
#define EV_EPS    1.e-6     // General epsilon for approximate equality
#define EV_EPS1   1.e-13    // Secondary epsilon for stricter comparisons
#define EV_EPSD   1.e-12    // Epsilon for convergence and stability checks

// Enumeration for matrix types
enum { 
    None = 0,   // Non-symmetric matrix
    SymMat = 1  // Symmetric matrix
};

/* 
 * Macro to check if two floating-point numbers are approximately equal.
 * It handles cases where either number is zero to avoid division by zero.
 */
#define EQUAL(x, y) ( \
    (((x) == 0.0) ? (fabs(y) < EV_EPS) : \
    (((y) == 0.0) ? (fabs(x) < EV_EPS) : \
    (fabs((x) - (y)) / (fabs(x) + fabs(y)) < EV_EPS))) \
)

/* Function Prototypes */

/**
 * @brief Computes the eigenvalues and eigenvectors of a 3x3 matrix.
 *
 * For symmetric matrices (SymMat = 1), the input array should contain
 * 6 elements in the order [a11, a12, a13, a22, a23, a33].
 *
 * For non-symmetric matrices (SymMat = 0), the input array should contain
 * 9 elements in row-major order [a11, a12, a13, a21, a22, a23, a31, a32, a33].
 *
 * @param symmat Indicator of matrix symmetry. Use SymMat (1) for symmetric matrices, None (0) otherwise.
 * @param mat Pointer to the input matrix. Size should be 6 for symmetric or 9 for non-symmetric matrices.
 * @param lambda Array to store the three eigenvalues. Must have space for at least 3 doubles.
 * @param v 3x3 matrix to store the eigenvectors as columns. Must be a 2D array with dimensions [3][3].
 * @return int 
 *         - Returns the number of real eigenvalues found (1, 2, or 3).
 *         - Returns 0 if no real eigenvalues are found (only applicable for non-symmetric matrices).
 *         - Returns -1 if the matrix type is unsupported or if an error occurs during computation.
 */
int eigen_3d(int symmat, double *mat, double lambda[3], double v[3][3]);

/**
 * @brief Computes the eigenvalues and eigenvectors of a 2x2 matrix.
 *
 * The input array should contain 4 elements in row-major order [a, b, c, d],
 * representing the matrix:
 * 
 * [ a  b ]
 * [ c  d ]
 *
 * Note: The function assumes real eigenvalues. If the matrix has complex eigenvalues,
 * the function will return 0, and eigenvalues/eigenvectors will not be computed.
 *
 * @param m Pointer to the input 2x2 matrix. Must contain 4 doubles.
 * @param l Array to store the two eigenvalues. Must have space for at least 2 doubles.
 * @param vp 2x2 matrix to store the eigenvectors as columns. Must be a 2D array with dimensions [2][2].
 * @return int 
 *         - Returns the number of real eigenvalues found (1 or 2).
 *         - Returns 0 if eigenvalues are complex.
 */
int eigen_2d(double *m, double *l, double vp[2][2]);

#endif // _EIGEN_H
