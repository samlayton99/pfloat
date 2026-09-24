/* pbit kernels, written once and compiled for the emulator (emul.c) and for native binary32 /
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
#include <stdatomic.h>
#include <stddef.h>
#include <stdlib.h>

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
typedef struct { pb_part_fn fn; void *ctx; long n, nchunks; atomic_long next; } pb_job;

static void pb_drain(pb_job *job) {
    for (;;) {
        long c = atomic_fetch_add(&job->next, 1);
        if (c >= job->nchunks) return;
        job->fn(job->ctx, job->n * c / job->nchunks, job->n * (c + 1) / job->nchunks);
    }
}

static void *pb_runner(void *arg) {
    pb_drain(arg);
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
int pb_set_threads(int n) {
    g_threads = n < 1 ? 1 : (n > PB_MAX_THREADS ? PB_MAX_THREADS : n);
    return g_threads;
}

#define PB_MAX_COEFFS 64

typedef struct {
    int p, emin, emax;
    T invln2, ln2hi, ln2lo, sat;                       /* tanh */
    T eps, prec, sfmin, huge, safmin, safmax;          /* DLAMCH / LA_CONSTANTS of the format */
    T tsml, tbig, ssml, sbig, epspow, hndrth;          /* DNRM2 thresholds, DBDSQR constants */
    int d;
    T a[PB_MAX_COEFFS];                                /* tanh Taylor coefficients */
} Consts;

/* Packed constants (doubles, each a format value; built by pbit/_formats.py):
 *  0 p, 1 emin, 2 emax, 3 INVLN2, 4 LN2HI, 5 LN2LO, 6 SAT, 7 eps, 8 prec, 9 sfmin, 10 huge,
 *  11 safmin, 12 safmax, 13 tsml, 14 tbig, 15 ssml, 16 sbig, 17 eps^(-1/8), 18 0.01, 19 d,
 *  20.. a_0 .. a_{d-1}. */
static int load_consts(Consts *k, const double *c) {
    k->p = (int)c[0]; k->emin = (int)c[1]; k->emax = (int)c[2];
    if (SETFMT(k->p, k->emin, k->emax)) return 1;
    k->invln2 = FROMD(c[3]); k->ln2hi = FROMD(c[4]); k->ln2lo = FROMD(c[5]); k->sat = FROMD(c[6]);
    k->eps = FROMD(c[7]); k->prec = FROMD(c[8]); k->sfmin = FROMD(c[9]); k->huge = FROMD(c[10]);
    k->safmin = FROMD(c[11]); k->safmax = FROMD(c[12]);
    k->tsml = FROMD(c[13]); k->tbig = FROMD(c[14]); k->ssml = FROMD(c[15]); k->sbig = FROMD(c[16]);
    k->epspow = FROMD(c[17]); k->hndrth = FROMD(c[18]);
    k->d = (int)c[19];
    if (k->d < 1 || k->d > PB_MAX_COEFFS) return 1;
    for (int j = 0; j < k->d; ++j) k->a[j] = FROMD(c[20 + j]);
    return 0;
}

#include "lapack_gelss.h"

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

typedef struct { const Consts *k; int op; const double *a, *b; double *out; size_t rows, len, kk, n; } ElemCtx;

static void binary_part(void *p, long lo, long hi) {
    ElemCtx *c = p;
    const double *a = c->a, *b = c->b;
    double *out = c->out;
    switch (c->op) {
        case 0: for (long i = lo; i < hi; ++i) out[i] = TOD(ADD(FROMD(a[i]), FROMD(b[i]))); break;
        case 1: for (long i = lo; i < hi; ++i) out[i] = TOD(SUB(FROMD(a[i]), FROMD(b[i]))); break;
        case 2: for (long i = lo; i < hi; ++i) out[i] = TOD(MUL(FROMD(a[i]), FROMD(b[i]))); break;
        case 3: for (long i = lo; i < hi; ++i) out[i] = TOD(DIV(FROMD(a[i]), FROMD(b[i]))); break;
    }
}

/* Elementwise binary op: 0 add, 1 subtract, 2 multiply, 3 divide. */
int pb_binary(const double *cst, int op, size_t n, const double *a, const double *b, double *out) {
    Consts k;
    ENTER();
    int bad = load_consts(&k, cst) || op < 0 || op > 3;
    if (!bad) {
        ElemCtx c = {&k, op, a, b, out, 0, 0, 0, 0};
        pb_parallel((long)n, 4, binary_part, &c);
    }
    LEAVE();
    return bad;
}

static void unary_part(void *p, long lo, long hi) {
    ElemCtx *c = p;
    if (c->op == 0) for (long i = lo; i < hi; ++i) c->out[i] = TOD(SQRT(FROMD(c->a[i])));
    else for (long i = lo; i < hi; ++i) c->out[i] = TOD(tanh_p(FROMD(c->a[i]), c->k));
}

/* Elementwise unary op: 0 sqrt, 1 tanh. */
int pb_unary(const double *cst, int op, size_t n, const double *a, double *out) {
    Consts k;
    ENTER();
    int bad = load_consts(&k, cst) || op < 0 || op > 1;
    if (!bad) {
        ElemCtx c = {&k, op, a, NULL, out, 0, 0, 0, 0};
        pb_parallel((long)n, op ? 64 : 4, unary_part, &c);
    }
    LEAVE();
    return bad;
}

static void sum_part(void *p, long lo, long hi) {
    ElemCtx *c = p;
    for (long r = lo; r < hi; ++r) {
        const double *row = c->a + (size_t)r * c->len;
        T s = FROMD(0.0);
        if (c->len > 0) s = FROMD(row[0]);
        for (size_t i = 1; i < c->len; ++i) s = ADD(s, FROMD(row[i]));
        c->out[r] = TOD(s);
    }
}

/* Row sums of a rows x len row-major array, left to right: s = x_0; s = s + x_i. */
int pb_sum(const double *cst, size_t rows, size_t len, const double *x, double *out) {
    Consts k;
    ENTER();
    int bad = load_consts(&k, cst);
    if (!bad) {
        ElemCtx c = {&k, 0, x, NULL, out, rows, len, 0, 0};
        pb_parallel((long)rows, (long)len + 1, sum_part, &c);
    }
    LEAVE();
    return bad;
}

static void matmul_part(void *p, long lo, long hi) {
    ElemCtx *c = p;
    for (long i = lo; i < hi; ++i)
        for (size_t j = 0; j < c->n; ++j) {
            T s = FROMD(0.0);
            for (size_t l = 0; l < c->kk; ++l)
                s = ADD(s, MUL(FROMD(c->a[(size_t)i * c->kk + l]), FROMD(c->b[l * c->n + j])));
            c->out[(size_t)i * c->n + j] = TOD(s);
        }
}

/* C = A B for row-major A (m x k), B (k x n): C_ij = 0, then C_ij = C_ij + A_il B_lj for
   l = 0..k-1, the operation order of reference BLAS DGEMM. */
int pb_matmul(const double *cst, size_t m, size_t kk, size_t n, const double *A, const double *B,
              double *C) {
    Consts k;
    ENTER();
    int bad = load_consts(&k, cst);
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
int pb_gelss(const double *cst, int m, int n, int nrhs, const double *A, const double *B, int ncut,
             const double *rconds, double *X, double *S, int *ranks) {
    Consts k;
    ENTER();
    if (load_consts(&k, cst)) { LEAVE(); return -10; }
    if (m < 0 || n < 0 || nrhs < 1 || ncut < 1) { LEAVE(); return -11; }
    int ldb = m > n ? m : n;
    size_t na = (size_t)m * n, nb = (size_t)(ldb > 0 ? ldb : 1) * nrhs, minmn = m < n ? m : n;
    T *a = malloc(sizeof(T) * (na ? na : 1)), *b = malloc(sizeof(T) * nb);
    T *s = malloc(sizeof(T) * (minmn ? minmn : 1)), *x = malloc(sizeof(T) * ((size_t)n * nrhs * ncut + 1));
    T *rc = malloc(sizeof(T) * ncut);
    int status = -4;
    if (a && b && s && x && rc) {
        for (size_t i = 0; i < na; ++i) a[i] = FROMD(A[i]);
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < ldb; ++i) b[(size_t)j * ldb + i] = i < m ? FROMD(B[(size_t)j * ldb + i]) : FROMD(0.0);
        for (int q = 0; q < ncut; ++q) rc[q] = FROMD(rconds[q]);
        status = lp_gelss(&k, m, n, nrhs, a, b, s, ncut, rc, x, ranks);
        if (status == 0) {
            for (size_t i = 0; i < minmn; ++i) S[i] = TOD(s[i]);
            for (size_t i = 0; i < (size_t)n * nrhs * ncut; ++i) X[i] = TOD(x[i]);
        }
    }
    free(a); free(b); free(s); free(x); free(rc);
    LEAVE();
    return status;
}
