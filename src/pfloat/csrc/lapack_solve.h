/* Derived from Reference LAPACK 3.12.1 (https://github.com/Reference-LAPACK/lapack).
 * Copyright (c) 1992-2023 The University of Tennessee and The University of Tennessee Research
 * Foundation; Copyright (c) 2000-2023 The University of California Berkeley; Copyright (c)
 * 2006-2023 The University of Colorado Denver. All rights reserved. Redistributed under LAPACK's
 * modified BSD license, reproduced in third_party/lapack-3.12.1/LICENSE.
 *
 * Reference LAPACK 3.12.1 DGESV (DGETRF -> DGETRF2, DGETRS) and DPOTRF (-> DPOTRF2) / DPOTRS,
 * with the reference BLAS they call (IDAMAX, DLASWP, DSCAL, DTRSM, DGEMM 'N','N', DSYRK 'L','N' and
 * 'U','T'), ported statement by statement to the rounding macros, as lapack_gelss.h. Unblocked
 * configuration: with ILAENV block size 1, DGETRF and DPOTRF call the recursive DGETRF2 and DPOTRF2
 * directly. Loops over independent columns (or rows) run on the thread pool; each entry keeps the
 * reference operation order.
 */

static int lp_idamax(int n, const T *dx, int incx) {
    if (n < 1 || incx <= 0) return 0;
    int best = 1;
    if (n == 1) return best;
    int ix = 1;
    T dmax = ABS(dx[0]);
    ix += incx;
    for (int i = 2; i <= n; ++i) {
        if (ABS(dx[ix - 1]) > dmax) {
            best = i;
            dmax = ABS(dx[ix - 1]);
        }
        ix += incx;
    }
    return best;
}

/* DLASWP: row interchanges k1..k2 (incx 1 forward, -1 backward) on n columns of a. */
static void lp_laswp(int n, T *a, int lda, int k1, int k2, const int *ipiv, int incx) {
    int i1, i2, inc, ix0;
    if (incx > 0) { ix0 = k1; i1 = k1; i2 = k2; inc = 1; }
    else if (incx < 0) { ix0 = k1 + (k1 - k2) * incx; i1 = k2; i2 = k1; inc = -1; }
    else return;
    int ix = ix0;
    for (int i = i1; inc > 0 ? i <= i2 : i >= i2; i += inc) {
        int ip = ipiv[ix - 1];
        if (ip != i)
            for (int k = 1; k <= n; ++k) {
                T temp = a[IX(i, k, lda)];
                a[IX(i, k, lda)] = a[IX(ip, k, lda)];
                a[IX(ip, k, lda)] = temp;
            }
        ix += incx;
    }
}

/* DGEMM with TRANSA = TRANSB = 'N', split over the columns j of C. */
typedef struct { int m, k; T alpha, beta; const T *a; int lda; const T *b; int ldb; T *c; int ldc; } GemmNNCtx;

static void gemm_nn_part(void *p, long lo, long hi) {
    GemmNNCtx *g = (GemmNNCtx *)p;
    const T zero = FROMD(0.0), one = FROMD(1.0);
    for (long j = lo + 1; j <= hi; ++j) {
        if (g->beta == zero) {
            for (int i = 1; i <= g->m; ++i) g->c[IX(i, j, g->ldc)] = zero;
        } else if (g->beta != one) {
            for (int i = 1; i <= g->m; ++i) g->c[IX(i, j, g->ldc)] = MUL(g->beta, g->c[IX(i, j, g->ldc)]);
        }
        for (int l = 1; l <= g->k; ++l) {
            T temp = MUL(g->alpha, g->b[IX(l, j, g->ldb)]);
            for (int i = 1; i <= g->m; ++i)
                g->c[IX(i, j, g->ldc)] = ADD(g->c[IX(i, j, g->ldc)], MUL(temp, g->a[IX(i, l, g->lda)]));
        }
    }
}

static void lp_gemm_nn(int m, int n, int k, T alpha, const T *a, int lda, const T *b, int ldb, T beta, T *c,
                       int ldc) {
    const T zero = FROMD(0.0), one = FROMD(1.0);
    if (m == 0 || n == 0 || ((alpha == zero || k == 0) && beta == one)) return;
    if (alpha == zero) {
        for (int j = 1; j <= n; ++j)
            for (int i = 1; i <= m; ++i)
                c[IX(i, j, ldc)] = beta == zero ? zero : MUL(beta, c[IX(i, j, ldc)]);
        return;
    }
    GemmNNCtx g = {m, k, alpha, beta, a, lda, b, ldb, c, ldc};
    pb_parallel(n, (long)m * k + 1, gemm_nn_part, &g);
}

/* DTRSM (all branches). Left side: split over the columns j of B; right side: over the rows i. */
typedef struct { char side, uplo, transa, diag; int m, n; T alpha; const T *a; int lda; T *b; int ldb; } TrsmCtx;

static void trsm_part(void *p, long lo, long hi) {
    TrsmCtx *t = (TrsmCtx *)p;
    const T zero = FROMD(0.0), one = FROMD(1.0);
    const T *a = t->a;
    T *b = t->b;
    int lda = t->lda, ldb = t->ldb, m = t->m, n = t->n;
    int nounit = t->diag == 'N', upper = t->uplo == 'U';
    T alpha = t->alpha;
    if (t->side == 'L') {
        for (long j = lo + 1; j <= hi; ++j) {
            if (t->transa == 'N') {
                if (alpha != one)
                    for (int i = 1; i <= m; ++i) b[IX(i, j, ldb)] = MUL(alpha, b[IX(i, j, ldb)]);
                if (upper) {
                    for (int k = m; k >= 1; --k)
                        if (b[IX(k, j, ldb)] != zero) {
                            if (nounit) b[IX(k, j, ldb)] = DIV(b[IX(k, j, ldb)], a[IX(k, k, lda)]);
                            for (int i = 1; i <= k - 1; ++i)
                                b[IX(i, j, ldb)] = SUB(b[IX(i, j, ldb)], MUL(b[IX(k, j, ldb)], a[IX(i, k, lda)]));
                        }
                } else {
                    for (int k = 1; k <= m; ++k)
                        if (b[IX(k, j, ldb)] != zero) {
                            if (nounit) b[IX(k, j, ldb)] = DIV(b[IX(k, j, ldb)], a[IX(k, k, lda)]);
                            for (int i = k + 1; i <= m; ++i)
                                b[IX(i, j, ldb)] = SUB(b[IX(i, j, ldb)], MUL(b[IX(k, j, ldb)], a[IX(i, k, lda)]));
                        }
                }
            } else {
                if (upper) {
                    for (int i = 1; i <= m; ++i) {
                        T temp = MUL(alpha, b[IX(i, j, ldb)]);
                        for (int k = 1; k <= i - 1; ++k) temp = SUB(temp, MUL(a[IX(k, i, lda)], b[IX(k, j, ldb)]));
                        if (nounit) temp = DIV(temp, a[IX(i, i, lda)]);
                        b[IX(i, j, ldb)] = temp;
                    }
                } else {
                    for (int i = m; i >= 1; --i) {
                        T temp = MUL(alpha, b[IX(i, j, ldb)]);
                        for (int k = i + 1; k <= m; ++k) temp = SUB(temp, MUL(a[IX(k, i, lda)], b[IX(k, j, ldb)]));
                        if (nounit) temp = DIV(temp, a[IX(i, i, lda)]);
                        b[IX(i, j, ldb)] = temp;
                    }
                }
            }
        }
    } else {
        /* rows lo+1..hi of B; the loops over j and k are those of the reference */
        long i0 = lo + 1, i1 = hi;
        if (t->transa == 'N') {
            if (upper) {
                for (int j = 1; j <= n; ++j) {
                    if (alpha != one)
                        for (long i = i0; i <= i1; ++i) b[IX(i, j, ldb)] = MUL(alpha, b[IX(i, j, ldb)]);
                    for (int k = 1; k <= j - 1; ++k)
                        if (a[IX(k, j, lda)] != zero)
                            for (long i = i0; i <= i1; ++i)
                                b[IX(i, j, ldb)] = SUB(b[IX(i, j, ldb)], MUL(a[IX(k, j, lda)], b[IX(i, k, ldb)]));
                    if (nounit) {
                        T temp = DIV(one, a[IX(j, j, lda)]);
                        for (long i = i0; i <= i1; ++i) b[IX(i, j, ldb)] = MUL(temp, b[IX(i, j, ldb)]);
                    }
                }
            } else {
                for (int j = n; j >= 1; --j) {
                    if (alpha != one)
                        for (long i = i0; i <= i1; ++i) b[IX(i, j, ldb)] = MUL(alpha, b[IX(i, j, ldb)]);
                    for (int k = j + 1; k <= n; ++k)
                        if (a[IX(k, j, lda)] != zero)
                            for (long i = i0; i <= i1; ++i)
                                b[IX(i, j, ldb)] = SUB(b[IX(i, j, ldb)], MUL(a[IX(k, j, lda)], b[IX(i, k, ldb)]));
                    if (nounit) {
                        T temp = DIV(one, a[IX(j, j, lda)]);
                        for (long i = i0; i <= i1; ++i) b[IX(i, j, ldb)] = MUL(temp, b[IX(i, j, ldb)]);
                    }
                }
            }
        } else {
            if (upper) {
                for (int k = n; k >= 1; --k) {
                    if (nounit) {
                        T temp = DIV(one, a[IX(k, k, lda)]);
                        for (long i = i0; i <= i1; ++i) b[IX(i, k, ldb)] = MUL(temp, b[IX(i, k, ldb)]);
                    }
                    for (int j = 1; j <= k - 1; ++j)
                        if (a[IX(j, k, lda)] != zero) {
                            T temp = a[IX(j, k, lda)];
                            for (long i = i0; i <= i1; ++i)
                                b[IX(i, j, ldb)] = SUB(b[IX(i, j, ldb)], MUL(temp, b[IX(i, k, ldb)]));
                        }
                    if (alpha != one)
                        for (long i = i0; i <= i1; ++i) b[IX(i, k, ldb)] = MUL(alpha, b[IX(i, k, ldb)]);
                }
            } else {
                for (int k = 1; k <= n; ++k) {
                    if (nounit) {
                        T temp = DIV(one, a[IX(k, k, lda)]);
                        for (long i = i0; i <= i1; ++i) b[IX(i, k, ldb)] = MUL(temp, b[IX(i, k, ldb)]);
                    }
                    for (int j = k + 1; j <= n; ++j)
                        if (a[IX(j, k, lda)] != zero) {
                            T temp = a[IX(j, k, lda)];
                            for (long i = i0; i <= i1; ++i)
                                b[IX(i, j, ldb)] = SUB(b[IX(i, j, ldb)], MUL(temp, b[IX(i, k, ldb)]));
                        }
                    if (alpha != one)
                        for (long i = i0; i <= i1; ++i) b[IX(i, k, ldb)] = MUL(alpha, b[IX(i, k, ldb)]);
                }
            }
        }
    }
}

static void lp_trsm(char side, char uplo, char transa, char diag, int m, int n, T alpha, const T *a, int lda,
                    T *b, int ldb) {
    const T zero = FROMD(0.0);
    if (m == 0 || n == 0) return;
    if (alpha == zero) {
        for (int j = 1; j <= n; ++j)
            for (int i = 1; i <= m; ++i) b[IX(i, j, ldb)] = zero;
        return;
    }
    TrsmCtx t = {side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb};
    if (side == 'L') pb_parallel(n, (long)m * m, trsm_part, &t);
    else pb_parallel(m, (long)n * n, trsm_part, &t);
}

/* DSYRK with UPLO = 'L', TRANS = 'N': C := alpha A A^T + beta C (lower triangle), split over j. */
typedef struct { int n, k; T alpha, beta; const T *a; int lda; T *c; int ldc; } SyrkCtx;

static void syrk_part(void *p, long lo, long hi) {
    SyrkCtx *s = (SyrkCtx *)p;
    const T zero = FROMD(0.0), one = FROMD(1.0);
    for (long j = lo + 1; j <= hi; ++j) {
        if (s->beta == zero) {
            for (int i = (int)j; i <= s->n; ++i) s->c[IX(i, j, s->ldc)] = zero;
        } else if (s->beta != one) {
            for (int i = (int)j; i <= s->n; ++i) s->c[IX(i, j, s->ldc)] = MUL(s->beta, s->c[IX(i, j, s->ldc)]);
        }
        for (int l = 1; l <= s->k; ++l)
            if (s->a[IX(j, l, s->lda)] != zero) {
                T temp = MUL(s->alpha, s->a[IX(j, l, s->lda)]);
                for (int i = (int)j; i <= s->n; ++i)
                    s->c[IX(i, j, s->ldc)] = ADD(s->c[IX(i, j, s->ldc)], MUL(temp, s->a[IX(i, l, s->lda)]));
            }
    }
}

static void lp_syrk_ln(int n, int k, T alpha, const T *a, int lda, T beta, T *c, int ldc) {
    const T zero = FROMD(0.0), one = FROMD(1.0);
    if (n == 0 || ((alpha == zero || k == 0) && beta == one)) return;
    if (alpha == zero) {
        for (int j = 1; j <= n; ++j)
            for (int i = j; i <= n; ++i) c[IX(i, j, ldc)] = beta == zero ? zero : MUL(beta, c[IX(i, j, ldc)]);
        return;
    }
    SyrkCtx s = {n, k, alpha, beta, a, lda, c, ldc};
    pb_parallel(n, (long)n * k + 1, syrk_part, &s);
}

/* DSYRK with UPLO = 'U', TRANS = 'T': C := alpha A^T A + beta C (upper triangle), split over j. */
static void syrk_ut_part(void *p, long lo, long hi) {
    SyrkCtx *s = (SyrkCtx *)p;
    const T zero = FROMD(0.0);
    for (long j = lo + 1; j <= hi; ++j)
        for (int i = 1; i <= (int)j; ++i) {
            T temp = zero;
            for (int l = 1; l <= s->k; ++l) temp = ADD(temp, MUL(s->a[IX(l, i, s->lda)], s->a[IX(l, j, s->lda)]));
            if (s->beta == zero) s->c[IX(i, j, s->ldc)] = MUL(s->alpha, temp);
            else s->c[IX(i, j, s->ldc)] = ADD(MUL(s->alpha, temp), MUL(s->beta, s->c[IX(i, j, s->ldc)]));
        }
}

static void lp_syrk_ut(int n, int k, T alpha, const T *a, int lda, T beta, T *c, int ldc) {
    const T zero = FROMD(0.0), one = FROMD(1.0);
    if (n == 0 || ((alpha == zero || k == 0) && beta == one)) return;
    if (alpha == zero) {
        for (int j = 1; j <= n; ++j)
            for (int i = 1; i <= j; ++i) c[IX(i, j, ldc)] = beta == zero ? zero : MUL(beta, c[IX(i, j, ldc)]);
        return;
    }
    SyrkCtx s = {n, k, alpha, beta, a, lda, c, ldc};
    pb_parallel(n, (long)n * k + 1, syrk_ut_part, &s);
}

/* DGETRF2: recursive LU with partial pivoting. ipiv is 1-based as in LAPACK. Returns INFO. */
static int lp_getrf2(const Consts *k, int m, int n, T *a, int lda, int *ipiv) {
    const T zero = FROMD(0.0), one = FROMD(1.0);
    int info = 0;
    if (m == 0 || n == 0) return 0;
    if (m == 1) {
        ipiv[0] = 1;
        if (a[IX(1, 1, lda)] == zero) info = 1;
    } else if (n == 1) {
        T sfmin = k->sfmin;
        int i = lp_idamax(m, a, 1);
        ipiv[0] = i;
        if (a[IX(i, 1, lda)] != zero) {
            if (i != 1) {
                T temp = a[IX(1, 1, lda)];
                a[IX(1, 1, lda)] = a[IX(i, 1, lda)];
                a[IX(i, 1, lda)] = temp;
            }
            if (ABS(a[IX(1, 1, lda)]) >= sfmin) {
                lp_scal(m - 1, DIV(one, a[IX(1, 1, lda)]), &a[IX(2, 1, lda)], 1);
            } else {
                for (int r = 1; r <= m - 1; ++r) a[IX(1 + r, 1, lda)] = DIV(a[IX(1 + r, 1, lda)], a[IX(1, 1, lda)]);
            }
        } else {
            info = 1;
        }
    } else {
        int n1 = (m < n ? m : n) / 2, n2 = n - n1;
        int iinfo = lp_getrf2(k, m, n1, a, lda, ipiv);
        if (info == 0 && iinfo > 0) info = iinfo;
        lp_laswp(n2, &a[IX(1, n1 + 1, lda)], lda, 1, n1, ipiv, 1);
        lp_trsm('L', 'L', 'N', 'U', n1, n2, one, a, lda, &a[IX(1, n1 + 1, lda)], lda);
        lp_gemm_nn(m - n1, n2, n1, NEG(one), &a[IX(n1 + 1, 1, lda)], lda, &a[IX(1, n1 + 1, lda)], lda, one,
                   &a[IX(n1 + 1, n1 + 1, lda)], lda);
        iinfo = lp_getrf2(k, m - n1, n2, &a[IX(n1 + 1, n1 + 1, lda)], lda, ipiv + n1);
        if (info == 0 && iinfo > 0) info = iinfo + n1;
        int mn = m < n ? m : n;
        for (int i = n1 + 1; i <= mn; ++i) ipiv[i - 1] += n1;
        lp_laswp(n1, a, lda, n1 + 1, mn, ipiv, 1);
    }
    return info;
}

/* DGETRS with TRANS = 'N'. */
static void lp_getrs(int n, int nrhs, const T *a, int lda, const int *ipiv, T *b, int ldb) {
    const T one = FROMD(1.0);
    if (n == 0 || nrhs == 0) return;
    lp_laswp(nrhs, b, ldb, 1, n, ipiv, 1);
    lp_trsm('L', 'L', 'N', 'U', n, nrhs, one, a, lda, b, ldb);
    lp_trsm('L', 'U', 'N', 'N', n, nrhs, one, a, lda, b, ldb);
}

/* DPOTRF2: recursive Cholesky, A = L L^T (lower) or U^T U (upper). Returns INFO. */
static int lp_potrf2(int upper, int n, T *a, int lda) {
    const T zero = FROMD(0.0), one = FROMD(1.0);
    if (n == 0) return 0;
    if (n == 1) {
        T a11 = a[IX(1, 1, lda)];
        if (a11 <= zero || a11 != a11) return 1;
        a[IX(1, 1, lda)] = SQRT(a11);
        return 0;
    }
    int n1 = n / 2, n2 = n - n1;
    int iinfo = lp_potrf2(upper, n1, a, lda);
    if (iinfo != 0) return iinfo;
    if (upper) {
        lp_trsm('L', 'U', 'T', 'N', n1, n2, one, a, lda, &a[IX(1, n1 + 1, lda)], lda);
        lp_syrk_ut(n2, n1, NEG(one), &a[IX(1, n1 + 1, lda)], lda, one, &a[IX(n1 + 1, n1 + 1, lda)], lda);
    } else {
        lp_trsm('R', 'L', 'T', 'N', n2, n1, one, a, lda, &a[IX(n1 + 1, 1, lda)], lda);
        lp_syrk_ln(n2, n1, NEG(one), &a[IX(n1 + 1, 1, lda)], lda, one, &a[IX(n1 + 1, n1 + 1, lda)], lda);
    }
    iinfo = lp_potrf2(upper, n2, &a[IX(n1 + 1, n1 + 1, lda)], lda);
    if (iinfo != 0) return iinfo + n1;
    return 0;
}

/* DPOTRS. */
static void lp_potrs(int upper, int n, int nrhs, const T *a, int lda, T *b, int ldb) {
    const T one = FROMD(1.0);
    if (n == 0 || nrhs == 0) return;
    if (upper) {
        lp_trsm('L', 'U', 'T', 'N', n, nrhs, one, a, lda, b, ldb);
        lp_trsm('L', 'U', 'N', 'N', n, nrhs, one, a, lda, b, ldb);
    } else {
        lp_trsm('L', 'L', 'N', 'N', n, nrhs, one, a, lda, b, ldb);
        lp_trsm('L', 'L', 'T', 'N', n, nrhs, one, a, lda, b, ldb);
    }
}
