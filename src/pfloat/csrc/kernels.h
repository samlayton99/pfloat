/* pfloat kernels, written once and compiled for the emulator (emul.c) and for native binary32 /
 * binary64 (native.c). The including file defines the number type T and
 *   ADD SUB MUL DIV SQRT  -- rounded arithmetic in the working format
 *   NEG ABS LDEXP RINT    -- exact operations
 *   FROMD(x)              -- a double that is already a format value, as T
 *   I2T(n)                -- an integer converted to the format (rounded, like Fortran DBLE)
 *   TOD(a)                -- T to double (exact)
 *   SETFMT(p, emin, emax) -- select the format (emulator) or check it (native types)
 * and every arithmetic step below goes through them.
 */
#include <pthread.h>
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
#include <atomic>
typedef std::atomic<long> pb_atomic_long;
#define PB_FETCH_ADD(p, v) std::atomic_fetch_add((p), (v))
#define PB_EXPORT extern "C"
#else
#include <stdatomic.h>
typedef atomic_long pb_atomic_long;
#define PB_FETCH_ADD(p, v) atomic_fetch_add((p), (v))
#define PB_EXPORT
#endif

/* The element type of the exported arrays: binary64 for the scalar builds (values of the format
   carried in doubles), T itself for the extended-precision build. */
#ifndef IO
#define IO double
#define LOADIO(x) FROMD(x)
#define STOREIO(x) TOD(x)
#endif
#ifndef THREAD_INIT
#define THREAD_INIT()
#endif

#ifndef THREAD_EXIT
#define THREAD_EXIT()
#endif
#ifndef ENTER
#define ENTER()
#define LEAVE()
#endif

/* ---------------------------------------------------------------- threads
 * pb_parallel runs fn over 0..n-1 in contiguous blocks shared among the threads (the caller works too).
 * Only loops whose items are independent use it, so every result keeps its operation order and
 * the output is bit-identical for any thread count. Small loops stay serial. */
static int g_threads = 1;
typedef void (*pb_part_fn)(void *ctx, long lo, long hi);
#define PB_MAX_THREADS 64
#ifndef PB_PAR_MIN_WORK
#define PB_PAR_MIN_WORK 65536L /* emulated operations; native builds set a larger threshold */
#endif
#define PB_CHUNKS_PER_THREAD 8

/* Blocks are claimed dynamically (an atomic counter), so faster cores take more of them. */
typedef struct { pb_part_fn fn; void *ctx; long n, nchunks; pb_atomic_long next; } pb_job;

static void pb_drain(pb_job *job) {
    for (;;) {
        long c = PB_FETCH_ADD(&job->next, 1);
        if (c >= job->nchunks) return;
        job->fn(job->ctx, job->n * c / job->nchunks, job->n * (c + 1) / job->nchunks);
    }
}

static void *pb_runner(void *arg) {
    THREAD_INIT();
    pb_drain((pb_job *)arg);
    THREAD_EXIT();
    return NULL;
}

static void pb_parallel(long n, long work_per_item, pb_part_fn fn, void *ctx) {
    int nt = g_threads < PB_MAX_THREADS ? g_threads : PB_MAX_THREADS;
    if (nt <= 1 || n < 2 || n * work_per_item < PB_PAR_MIN_WORK) { fn(ctx, 0, n); return; }
    if (nt > n) nt = (int)n;
    long nchunks = (long)nt * PB_CHUNKS_PER_THREAD;
    if (nchunks > n) nchunks = n;
    pb_job job = {fn, ctx, n, nchunks, 0};
    pthread_t th[PB_MAX_THREADS];
    int started[PB_MAX_THREADS];
    for (int t = 1; t < nt; ++t) started[t] = pthread_create(&th[t], NULL, pb_runner, &job) == 0;
    pb_drain(&job);
    for (int t = 1; t < nt; ++t)
        if (started[t]) pthread_join(th[t], NULL);
}

/* Threads used by the kernels (1 = serial); returns the value set. */
PB_EXPORT int pb_set_threads(int n) {
    g_threads = n < 1 ? 1 : (n > PB_MAX_THREADS ? PB_MAX_THREADS : n);
    return g_threads;
}

#define PB_MAX_COEFFS 640

typedef struct {
    int p, emin, emax;
    T invln2, ln2hi, ln2lo, sat;                       /* tanh */
    T eps, prec, sfmin, huge, safmin, safmax;          /* DLAMCH / LA_CONSTANTS of the format */
    T tsml, tbig, ssml, sbig, epspow, hndrth;          /* DNRM2 thresholds, DBDSQR constants */
    int d;
    T a[PB_MAX_COEFFS];                                /* tanh Taylor coefficients */
} Consts;

/* Packed constants: c holds them as doubles (the integers p, emin, emax, d are read from it) and
 * v as values of the format (for the scalar builds v == c; built by pfloat/_formats.py):
 *  0 p, 1 emin, 2 emax, 3 INVLN2, 4 LN2HI, 5 LN2LO, 6 SAT, 7 eps, 8 prec, 9 sfmin, 10 huge,
 *  11 safmin, 12 safmax, 13 tsml, 14 tbig, 15 ssml, 16 sbig, 17 eps^(-1/8), 18 0.01, 19 d,
 *  20.. a_0 .. a_{d-1}. */
static int load_consts(Consts *k, const double *c, const IO *v) {
    k->p = (int)c[0]; k->emin = (int)c[1]; k->emax = (int)c[2];
    if (SETFMT(k->p, k->emin, k->emax)) return 1;
    k->invln2 = LOADIO(v[3]); k->ln2hi = LOADIO(v[4]); k->ln2lo = LOADIO(v[5]); k->sat = LOADIO(v[6]);
    k->eps = LOADIO(v[7]); k->prec = LOADIO(v[8]); k->sfmin = LOADIO(v[9]); k->huge = LOADIO(v[10]);
    k->safmin = LOADIO(v[11]); k->safmax = LOADIO(v[12]);
    k->tsml = LOADIO(v[13]); k->tbig = LOADIO(v[14]); k->ssml = LOADIO(v[15]); k->sbig = LOADIO(v[16]);
    k->epspow = LOADIO(v[17]); k->hndrth = LOADIO(v[18]);
    k->d = (int)c[19];
    if (k->d < 1 || k->d > PB_MAX_COEFFS) return 1;
    for (int j = 0; j < k->d; ++j) k->a[j] = LOADIO(v[20 + j]);
    return 0;
}

#include "lapack_gelss.h"
#include "lapack_solve.h"

/* tanh from the five operations: tanh|z| = -E / (E + 2), E = expm1(-2|z|) by Cody-Waite
   reduction and a Horner Taylor polynomial whose degree is set by p. */
static T tanh_p(T z, const Consts *k) {
    const T one = FROMD(1.0), two = FROMD(2.0), zero = FROMD(0.0);
    if (z != z) return z;
    T a = ABS(z);
    if (a >= k->sat) return z < zero ? NEG(one) : one;
    T u = NEG(LDEXP(a, 1));
    T kf = RINT(MUL(u, k->invln2));
    int kk = (int)TOD(kf);
    T r = u;
    if (kk != 0) r = SUB(SUB(u, MUL(kf, k->ln2hi)), MUL(kf, k->ln2lo));
    T q = k->a[k->d - 1];
    for (int j = k->d - 2; j >= 0; --j) q = ADD(k->a[j], MUL(r, q));
    T P = MUL(r, q);
    T E = P;
    if (kk != 0) E = ADD(LDEXP(P, kk), SUB(LDEXP(one, kk), one));
    T t = NEG(DIV(E, ADD(E, two)));
    return z < zero ? NEG(t) : t;
}

typedef struct { const Consts *k; int op; const IO *a, *b; IO *out; size_t rows, len, kk, n; } ElemCtx;

static void binary_part(void *p, long lo, long hi) {
    ElemCtx *c = (ElemCtx *)p;
    const IO *a = c->a, *b = c->b;
    IO *out = c->out;
    switch (c->op) {
        case 0: for (long i = lo; i < hi; ++i) out[i] = STOREIO(ADD(LOADIO(a[i]), LOADIO(b[i]))); break;
        case 1: for (long i = lo; i < hi; ++i) out[i] = STOREIO(SUB(LOADIO(a[i]), LOADIO(b[i]))); break;
        case 2: for (long i = lo; i < hi; ++i) out[i] = STOREIO(MUL(LOADIO(a[i]), LOADIO(b[i]))); break;
        case 3: for (long i = lo; i < hi; ++i) out[i] = STOREIO(DIV(LOADIO(a[i]), LOADIO(b[i]))); break;
    }
}

/* Elementwise binary op: 0 add, 1 subtract, 2 multiply, 3 divide. */
PB_EXPORT int pb_binary(const double *cst, const IO *cv, int op, size_t n, const IO *a, const IO *b, IO *out) {
    Consts k;
    ENTER();
    int bad = load_consts(&k, cst, cv) || op < 0 || op > 3;
    if (!bad) {
        ElemCtx c = {&k, op, a, b, out, 0, 0, 0, 0};
        pb_parallel((long)n, 4, binary_part, &c);
    }
    LEAVE();
    return bad;
}

static void unary_part(void *p, long lo, long hi) {
    ElemCtx *c = (ElemCtx *)p;
    if (c->op == 0) for (long i = lo; i < hi; ++i) c->out[i] = STOREIO(SQRT(LOADIO(c->a[i])));
    else for (long i = lo; i < hi; ++i) c->out[i] = STOREIO(tanh_p(LOADIO(c->a[i]), c->k));
}

/* Elementwise unary op: 0 sqrt, 1 tanh. */
PB_EXPORT int pb_unary(const double *cst, const IO *cv, int op, size_t n, const IO *a, IO *out) {
    Consts k;
    ENTER();
    int bad = load_consts(&k, cst, cv) || op < 0 || op > 1;
    if (!bad) {
        ElemCtx c = {&k, op, a, NULL, out, 0, 0, 0, 0};
        pb_parallel((long)n, op ? 64 : 4, unary_part, &c);
    }
    LEAVE();
    return bad;
}

static void sum_part(void *p, long lo, long hi) {
    ElemCtx *c = (ElemCtx *)p;
    for (long r = lo; r < hi; ++r) {
        const IO *row = c->a + (size_t)r * c->len;
        T s = FROMD(0.0);
        if (c->len > 0) s = LOADIO(row[0]);
        for (size_t i = 1; i < c->len; ++i) s = ADD(s, LOADIO(row[i]));
        c->out[r] = STOREIO(s);
    }
}

/* Row sums of a rows x len row-major array, left to right: s = x_0; s = s + x_i. */
PB_EXPORT int pb_sum(const double *cst, const IO *cv, size_t rows, size_t len, const IO *x, IO *out) {
    Consts k;
    ENTER();
    int bad = load_consts(&k, cst, cv);
    if (!bad) {
        ElemCtx c = {&k, 0, x, NULL, out, rows, len, 0, 0};
        pb_parallel((long)rows, (long)len + 1, sum_part, &c);
    }
    LEAVE();
    return bad;
}

/* Rows lo..hi of C. Each C_ij starts at 0 and adds A_il B_lj for l = 0, 1, ..., k-1 in order;
   the loops run over l then j so that B is read by rows. */
static void matmul_part(void *p, long lo, long hi) {
    ElemCtx *c = (ElemCtx *)p;
    T *acc = (T *)malloc(sizeof(T) * (c->n ? c->n : 1));
    if (!acc) return;
    for (long i = lo; i < hi; ++i) {
        for (size_t j = 0; j < c->n; ++j) acc[j] = FROMD(0.0);
        for (size_t l = 0; l < c->kk; ++l) {
            T ail = LOADIO(c->a[(size_t)i * c->kk + l]);
            const IO *brow = c->b + l * c->n;
            for (size_t j = 0; j < c->n; ++j) acc[j] = ADD(acc[j], MUL(ail, LOADIO(brow[j])));
        }
        for (size_t j = 0; j < c->n; ++j) c->out[(size_t)i * c->n + j] = STOREIO(acc[j]);
    }
    free(acc);
}

/* C = A B for row-major A (m x k), B (k x n): C_ij = 0, then C_ij = C_ij + A_il B_lj for
   l = 0..k-1, the operation order of reference BLAS DGEMM. */
PB_EXPORT int pb_matmul(const double *cst, const IO *cv, size_t m, size_t kk, size_t n, const IO *A,
                        const IO *B, IO *C) {
    Consts k;
    ENTER();
    int bad = load_consts(&k, cst, cv);
    if (!bad) {
        ElemCtx c = {&k, 0, A, B, C, m, 0, kk, n};
        pb_parallel((long)m, (long)(kk * n) + 1, matmul_part, &c);
    }
    LEAVE();
    return bad;
}

/* Least squares by the DGELSS port. A: m x n column-major; B: ldb x nrhs column-major with
 * ldb = max(m, n) (rows m.. are ignored on input). X: ncut blocks of n x nrhs column-major;
 * S: min(m, n); ranks: ncut. Returns DGELSS INFO (0 ok, > 0 no convergence) or < 0 on error. */
PB_EXPORT int pb_gelss(const double *cst, const IO *cv, int m, int n, int nrhs, const IO *A, const IO *B,
                       int ncut, const IO *rconds, IO *X, IO *S, int *ranks) {
    Consts k;
    ENTER();
    if (load_consts(&k, cst, cv)) { LEAVE(); return -10; }
    if (m < 0 || n < 0 || nrhs < 1 || ncut < 1) { LEAVE(); return -11; }
    int ldb = m > n ? m : n;
    size_t na = (size_t)m * n, nb = (size_t)(ldb > 0 ? ldb : 1) * nrhs, minmn = m < n ? m : n;
    T *a = (T *)malloc(sizeof(T) * (na ? na : 1)), *b = (T *)malloc(sizeof(T) * nb);
    T *s = (T *)malloc(sizeof(T) * (minmn ? minmn : 1));
    T *x = (T *)malloc(sizeof(T) * ((size_t)n * nrhs * ncut + 1));
    T *rc = (T *)malloc(sizeof(T) * ncut);
    int status = -4;
    if (a && b && s && x && rc) {
        for (size_t i = 0; i < na; ++i) a[i] = LOADIO(A[i]);
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < ldb; ++i) b[(size_t)j * ldb + i] = i < m ? LOADIO(B[(size_t)j * ldb + i]) : FROMD(0.0);
        for (int q = 0; q < ncut; ++q) rc[q] = LOADIO(rconds[q]);
        status = lp_gelss(&k, m, n, nrhs, a, b, s, ncut, rc, x, ranks);
        if (status == 0) {
            for (size_t i = 0; i < minmn; ++i) S[i] = STOREIO(s[i]);
            for (size_t i = 0; i < (size_t)n * nrhs * ncut; ++i) X[i] = STOREIO(x[i]);
        }
    }
    free(a); free(b); free(s); free(x); free(rc);
    LEAVE();
    return status;
}

static T *load_matrix(const IO *src, size_t n) {
    T *a = (T *)malloc(sizeof(T) * (n ? n : 1));
    if (a)
        for (size_t i = 0; i < n; ++i) a[i] = LOADIO(src[i]);
    return a;
}

static void store_matrix(IO *dst, const T *a, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = STOREIO(a[i]);
}

/* LU with partial pivoting by the DGETRF2 port (DGETRF at block size 1). A: m x n column-major;
 * LU receives the factors in the same layout, ipiv the min(m, n) 1-based row interchanges.
 * Returns DGETRF INFO (0, or i > 0 when U(i,i) is exactly zero) or < 0 on error. */
PB_EXPORT int pb_getrf(const double *cst, const IO *cv, int m, int n, const IO *A, IO *LU, int *ipiv) {
    Consts k;
    ENTER();
    if (load_consts(&k, cst, cv)) { LEAVE(); return -10; }
    if (m < 0 || n < 0) { LEAVE(); return -11; }
    size_t na = (size_t)m * n;
    T *a = load_matrix(A, na);
    int info = -4;
    if (a) {
        info = lp_getrf2(&k, m, n, a, m > 1 ? m : 1, ipiv);
        store_matrix(LU, a, na);
    }
    free(a);
    LEAVE();
    return info;
}

/* DGETRS with TRANS = 'N': X = A^-1 B from the factors of pb_getrf (n x n) and B (n x nrhs,
 * column-major). */
PB_EXPORT int pb_getrs(const double *cst, const IO *cv, int n, int nrhs, const IO *LU, const int *ipiv,
                       const IO *B, IO *X) {
    Consts k;
    ENTER();
    if (load_consts(&k, cst, cv)) { LEAVE(); return -10; }
    if (n < 0 || nrhs < 0) { LEAVE(); return -11; }
    size_t na = (size_t)n * n, nb = (size_t)n * nrhs;
    T *a = load_matrix(LU, na), *b = load_matrix(B, nb);
    int info = -4;
    if (a && b) {
        lp_getrs(n, nrhs, a, n > 1 ? n : 1, ipiv, b, n > 1 ? n : 1);
        store_matrix(X, b, nb);
        info = 0;
    }
    free(a); free(b);
    LEAVE();
    return info;
}

/* Cholesky by the DPOTRF2 port (DPOTRF at block size 1). A: n x n column-major; only the lower
 * (upper = 0) or upper triangle is read. C receives the array as DPOTRF leaves it: the factor in
 * that triangle, the other triangle untouched. Returns DPOTRF INFO (0, or i > 0 when the leading
 * minor of order i is not positive definite) or < 0 on error. */
PB_EXPORT int pb_potrf(const double *cst, const IO *cv, int upper, int n, const IO *A, IO *C) {
    Consts k;
    ENTER();
    if (load_consts(&k, cst, cv)) { LEAVE(); return -10; }
    if (n < 0) { LEAVE(); return -11; }
    size_t na = (size_t)n * n;
    T *a = load_matrix(A, na);
    int info = -4;
    if (a) {
        info = lp_potrf2(upper, n, a, n > 1 ? n : 1);
        store_matrix(C, a, na);
    }
    free(a);
    LEAVE();
    return info;
}

/* DPOTRS: X = A^-1 B from the factor of pb_potrf. */
PB_EXPORT int pb_potrs(const double *cst, const IO *cv, int upper, int n, int nrhs, const IO *C, const IO *B,
                       IO *X) {
    Consts k;
    ENTER();
    if (load_consts(&k, cst, cv)) { LEAVE(); return -10; }
    if (n < 0 || nrhs < 0) { LEAVE(); return -11; }
    size_t na = (size_t)n * n, nb = (size_t)n * nrhs;
    T *a = load_matrix(C, na), *b = load_matrix(B, nb);
    int info = -4;
    if (a && b) {
        lp_potrs(upper, n, nrhs, a, n > 1 ? n : 1, b, n > 1 ? n : 1);
        store_matrix(X, b, nb);
        info = 0;
    }
    free(a); free(b);
    LEAVE();
    return info;
}

static void nrm2_part(void *p, long lo, long hi) {
    ElemCtx *c = (ElemCtx *)p;
    T *buf = (T *)malloc(sizeof(T) * (c->len ? c->len : 1));
    for (long r = lo; r < hi; ++r) {
        const IO *row = c->a + (size_t)r * c->len;
        for (size_t i = 0; i < c->len; ++i) buf[i] = LOADIO(row[i]);
        c->out[r] = STOREIO(lp_nrm2(c->k, (int)c->len, buf, 1));
    }
    free(buf);
}

/* Euclidean norms of the rows of a rows x len row-major array by the DNRM2 port (Blue's scaled
 * sums). */
PB_EXPORT int pb_nrm2(const double *cst, const IO *cv, size_t rows, size_t len, const IO *x, IO *out) {
    Consts k;
    ENTER();
    int bad = load_consts(&k, cst, cv);
    if (!bad) {
        ElemCtx c = {&k, 0, x, NULL, out, rows, len, 0, 0};
        pb_parallel((long)rows, 4 * (long)len + 1, nrm2_part, &c);
    }
    LEAVE();
    return bad;
}
