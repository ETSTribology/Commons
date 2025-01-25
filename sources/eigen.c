// eigen.c
#include "eigen.h"
#include <math.h>
#include <string.h>
#include <omp.h>     // OpenMP header for parallelization

// Define constants
#define EV_MAXIT    50
// EV_EPSD is defined in the header as 1.0e-12

// Identity matrix for initialization
static double Id[3][3] = { 
    {1.0, 0.0, 0.0}, 
    {0.0, 1.0, 0.0}, 
    {0.0, 0.0, 1.0} 
};

/* 
 * Find roots of the cubic polynomial: P(x) = x^3 + b x^2 + c x + d 
 * Parameters:
 *   p[4] - coefficients of the polynomial, where p[0] = d, p[1] = c, p[2] = b, p[3] = 1.0
 *   x[3] - array to store the roots
 * Returns:
 *   Number of real roots found (1, 2, or 3)
 */
static int newton3(double p[4], double x[3]) {
    double b = p[2], c = p[1], d = p[0];
    double da = 3.0, db = 2.0 * b;
    double delta = db * db - 12.0 * c;  // Discriminant of derivative f'(x) = 3x^2 + 2bx + c
    double epsd = EV_EPSD * (db * db + 12.0 * fabs(c));

    // Check for multiple roots via derivative's roots
    if (delta > epsd) {
        double sqrt_delta = sqrt(delta);
        double dx0 = (-db + sqrt_delta) / (2.0 * da);  // f'(x) = 0 => x = (-2b ± sqrt(delta)) / 6
        double dx1 = (-db - sqrt_delta) / (2.0 * da);
        double fdx0 = d + dx0 * (c + dx0 * (b + dx0));
        double fdx1 = d + dx1 * (c + dx1 * (b + dx1));

        if (fabs(fdx0) < EV_EPSD) {
            // dx0 is a root with multiplicity
            x[0] = dx0;
            x[1] = dx0;
            x[2] = -b - 2.0 * dx0;
            return (fabs(x[2] - dx0) < EV_EPSD) ? 1 : 2;
        } 
        if (fabs(fdx1) < EV_EPSD) {
            // dx1 is a root with multiplicity
            x[0] = dx1;
            x[1] = dx1;
            x[2] = -b - 2.0 * dx1;
            return (fabs(x[2] - dx1) < EV_EPSD) ? 1 : 2;
        }
    } else if (fabs(delta) <= epsd) {
        // Triple root or single root with double derivative root
        double triple_root = -b / 3.0;
        x[0] = x[1] = x[2] = triple_root;
        return 1;
    }

    // Newton-Raphson to find a single real root
    double x1 = -b / 3.0;  // Initial guess (inflection point)
    int it;
    for (it = 0; it < EV_MAXIT; ++it) {
        double fx = d + x1 * (c + x1 * (b + x1));
        double dfx = c + x1 * (2.0 * b + 3.0 * x1);
        if (fabs(dfx) < EV_EPSD) break; // Avoid division by zero
        double x2 = x1 - fx / dfx;
        if (fabs(x2 - x1) <= EV_EPSD * (fabs(x2) + 1.0)) {
            x1 = x2;
            break;
        }
        x1 = x2;
    }
    x[0] = x1;

    // Deflate to quadratic polynomial: P(x) = (x - x1)(x^2 + bb * x + cc)
    double bb = b + x1;
    double cc = c + x1 * bb;
    double disc = bb * bb - 4.0 * cc;

    if (disc < -EV_EPSD) {
        // One real root
        return 1;
    } else if (disc < EV_EPSD) {
        // Two real roots (double root)
        x[1] = x[2] = -0.5 * bb;
        return 2;
    } else {
        // Three real roots
        double sqrt_disc = sqrt(disc);
        x[1] = 0.5 * (-bb + sqrt_disc);
        x[2] = 0.5 * (-bb - sqrt_disc);
        return 3;
    }
}

/* 
 * Find eigenvalues and eigenvectors of a 3x3 matrix.
 * Parameters:
 *   type - matrix type (SymMat for symmetric matrices, None otherwise)
 *   m - input matrix (array of 6 elements for symmetric or 9 for general)
 *   l - output eigenvalues (array of 3 elements)
 *   v - output eigenvectors (3x3 matrix)
 * Returns:
 *   Number of real eigenvalues found (1, 2, or 3). Returns 0 on failure.
 */
int eigen_3d(int type, double *m, double *l, double v[3][3]) {
    double a11, a12, a13, a22, a23, a33;
    double p[4];
    double maxm = 0.0;
    int k, n;

    // Initialize eigenvectors to identity
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            v[i][j] = Id[i][j];

    if (type == SymMat) {
        // For symmetric matrices, only 6 unique elements are provided
        a11 = m[0]; a12 = m[1]; a13 = m[2];
        a22 = m[3]; a23 = m[4];
        a33 = m[5];

        // Normalize matrix to avoid overflow
        maxm = fabs(a11);
        if (fabs(a12) > maxm) maxm = fabs(a12);
        if (fabs(a13) > maxm) maxm = fabs(a13);
        if (fabs(a22) > maxm) maxm = fabs(a22);
        if (fabs(a23) > maxm) maxm = fabs(a23);
        if (fabs(a33) > maxm) maxm = fabs(a33);

        if (maxm > EV_EPSD) {
            a11 /= maxm;
            a12 /= maxm;
            a13 /= maxm;
            a22 /= maxm;
            a23 /= maxm;
            a33 /= maxm;
        }

        // Check if the matrix is diagonal
        double maxd = fabs(a12);
        maxd = fmax(maxd, fabs(a13));
        maxd = fmax(maxd, fabs(a23));

        if (maxd < EV_EPSD) {
            // Diagonal matrix: eigenvalues are the diagonal elements
            l[0] = a11 * maxm;
            l[1] = a22 * maxm;
            l[2] = a33 * maxm;
            // Eigenvectors are already set to identity
            return 3;
        }

        // Build characteristic polynomial: P(X) = X^3 - trace(X) X^2 + (sum of principal minors) X - det(A) = 0
        double trace = a11 + a22 + a33;
        double minor_sum = a11 * a22 + a11 * a33 + a22 * a33 - a12 * a12 - a13 * a13 - a23 * a23;
        double det = a11 * (a22 * a33 - a23 * a23) - a12 * (a12 * a33 - a13 * a23) + a13 * (a12 * a23 - a22 * a13);

        p[3] = 1.0;
        p[2] = -trace;
        p[1] = minor_sum;
        p[0] = -det;
    } else {
        // For general (non-symmetric) matrices, handle all 9 elements
        double a21, a31, a32;
        a11 = m[0]; a12 = m[1]; a13 = m[2];
        a21 = m[3]; a22 = m[4]; a23 = m[5];
        a31 = m[6]; a32 = m[7]; a33 = m[8];

        // Normalize matrix to avoid overflow
        maxm = fabs(a11);
        for (k = 1; k < 9; k++) {
            double valm = fabs(m[k]);
            if (valm > maxm) maxm = valm;
        }

        if (maxm > EV_EPSD) {
            a11 /= maxm; a12 /= maxm; a13 /= maxm;
            a21 /= maxm; a22 /= maxm; a23 /= maxm;
            a31 /= maxm; a32 /= maxm; a33 /= maxm;
        }

        // Check if the matrix is diagonal
        double maxd = fabs(a12);
        maxd = fmax(maxd, fabs(a13));
        maxd = fmax(maxd, fabs(a23));
        maxd = fmax(maxd, fabs(a21));
        maxd = fmax(maxd, fabs(a31));
        maxd = fmax(maxd, fabs(a32));

        if (maxd < EV_EPSD) {
            // Diagonal matrix: eigenvalues are the diagonal elements
            l[0] = a11 * maxm;
            l[1] = a22 * maxm;
            l[2] = a33 * maxm;
            // Eigenvectors are already set to identity
            return 3;
        }

        // Build characteristic polynomial for general matrix
        double trace = a11 + a22 + a33;
        double minor_sum = a11 * a22 + a11 * a33 + a22 * a33 - a12 * a21 - a13 * a31 - a23 * a32;
        double det = a11 * (a22 * a33 - a23 * a32) 
                   - a12 * (a21 * a33 - a23 * a31) 
                   + a13 * (a21 * a32 - a22 * a31);

        p[3] = 1.0;
        p[2] = -trace;
        p[1] = minor_sum;
        p[0] = -det;
    }

    // Solve the characteristic polynomial to find eigenvalues
    n = newton3(p, l);
    if (n <= 0) return 0;

    // Scale eigenvalues back
    #pragma omp simd
    for (k = 0; k < n; ++k) {
        l[k] *= maxm;
    }

    // Eigenvector computation for symmetric matrices
    if (type == SymMat) {
        // Parallelize over each eigenvalue to compute eigenvectors
        #pragma omp parallel for
        for (k = 0; k < n; ++k) {
            double lambda = l[k];
            // Form the matrix (A - lambda * I)
            double row1[3] = { a11 - (lambda / maxm), a12, a13 };
            double row2[3] = { a12, a22 - (lambda / maxm), a23 };
            double row3[3] = { a13, a23, a33 - (lambda / maxm) };

            // Compute cross products of rows to find eigenvectors
            double vx1[3], vx2[3], vx3[3];
            double dd1, dd2, dd3;

            // Cross product row1 x row2
            vx1[0] = row1[1] * row2[2] - row1[2] * row2[1];
            vx1[1] = row1[2] * row2[0] - row1[0] * row2[2];
            vx1[2] = row1[0] * row2[1] - row1[1] * row2[0];
            dd1 = vx1[0] * vx1[0] + vx1[1] * vx1[1] + vx1[2] * vx1[2];

            // Cross product row1 x row3
            vx2[0] = row1[1] * row3[2] - row1[2] * row3[1];
            vx2[1] = row1[2] * row3[0] - row1[0] * row3[2];
            vx2[2] = row1[0] * row3[1] - row1[1] * row3[0];
            dd2 = vx2[0] * vx2[0] + vx2[1] * vx2[1] + vx2[2] * vx2[2];

            // Cross product row2 x row3
            vx3[0] = row2[1] * row3[2] - row2[2] * row3[1];
            vx3[1] = row2[2] * row3[0] - row2[0] * row3[2];
            vx3[2] = row2[0] * row3[1] - row2[1] * row3[0];
            dd3 = vx3[0] * vx3[0] + vx3[1] * vx3[1] + vx3[2] * vx3[2];

            // Select the cross product with the largest magnitude to avoid numerical instability
            double *selected_vx;
            double selected_dd;
            if (dd1 > dd2 && dd1 > dd3) {
                selected_vx = vx1;
                selected_dd = dd1;
            } else if (dd2 > dd3) {
                selected_vx = vx2;
                selected_dd = dd2;
            } else {
                selected_vx = vx3;
                selected_dd = dd3;
            }

            if (selected_dd > EV_EPSD) {
                double inv_norm = 1.0 / sqrt(selected_dd);
                v[k][0] = selected_vx[0] * inv_norm;
                v[k][1] = selected_vx[1] * inv_norm;
                v[k][2] = selected_vx[2] * inv_norm;
            } else {
                // If all cross products are near zero, default to identity
                v[k][0] = (k == 0) ? 1.0 : 0.0;
                v[k][1] = (k == 1) ? 1.0 : 0.0;
                v[k][2] = (k == 2) ? 1.0 : 0.0;
            }
        }
    } else {
        // For general (non-symmetric) matrices, eigenvectors are not guaranteed to be orthogonal
        // Implementing eigenvector computation for non-symmetric matrices is beyond this scope
        // Typically, numerical libraries like LAPACK are used for this purpose
        // Here, we return the identity matrix as a placeholder
    }

    return n;
}

/* 
 * Find eigenvalues and eigenvectors of a 2x2 matrix.
 * Parameters:
 *   m - input matrix (array of 4 elements: [a, b, c, d])
 *   l - output eigenvalues (array of 2 elements)
 *   vp - output eigenvectors (2x2 matrix)
 * Returns:
 *   Number of real eigenvalues found (1 or 2). Returns 0 if eigenvalues are complex.
 */
int eigen_2d(double *m, double *l, double vp[2][2]) {
    double a = m[0], b = m[1], c = m[2], d = m[3];
    vp[0][0] = 1.0; vp[0][1] = 0.0;
    vp[1][0] = 0.0; vp[1][1] = 1.0;

    // Check if the matrix is diagonal
    if (fabs(b) < EV_EPSD && fabs(c) < EV_EPSD) {
        l[0] = a;
        l[1] = d;
        return 2;
    }

    // Compute trace and determinant
    double trace = a + d;
    double det = a * d - b * c;
    double disc = trace * trace - 4.0 * det;

    if (disc < -EV_EPSD) {
        // Complex eigenvalues (not handled)
        return 0;
    }

    disc = (disc < 0.0) ? 0.0 : sqrt(disc);
    l[0] = 0.5 * (trace + disc);
    l[1] = 0.5 * (trace - disc);

    // Compute eigenvectors
    #pragma omp parallel for
    for (int k = 0; k < 2; ++k) {
        if (fabs(b) > EV_EPSD || fabs(c) > EV_EPSD) {
            if (fabs(b) > fabs(c)) {
                vp[k][0] = l[k] - d;
                vp[k][1] = b;
            } else {
                vp[k][0] = c;
                vp[k][1] = l[k] - a;
            }
        } else {
            // Default to identity if unable to compute
            vp[k][0] = (k == 0) ? 1.0 : 0.0;
            vp[k][1] = (k == 1) ? 1.0 : 0.0;
        }

        // Normalize eigenvectors using SIMD
        double norm = sqrt(vp[k][0] * vp[k][0] + vp[k][1] * vp[k][1]);
        if (norm > EV_EPSD) {
            #pragma omp simd
            for (int i = 0; i < 2; ++i) {
                vp[k][i] /= norm;
            }
        } else {
            // If the vector is zero, assign default
            vp[k][0] = (k == 0) ? 1.0 : 0.0;
            vp[k][1] = (k == 1) ? 1.0 : 0.0;
        }
    }

    return 2;
}
