/* pfloat extended precision: the same kernels and LAPACK port on formats with p > 53 bits.
 *
 * The number type T is a fixed-size value (kind, sign, exponent, significand limbs) on which
 * MPFR works in place through its custom interface. Every +, -, *, /, sqrt is one MPFR operation
 * rounded to nearest (ties to even) at precision p, followed by mpfr_subnormalize with the
 * format's exponent range set, so results are those of an IEEE-style binary format with a p-bit
 * significand and gradual underflow (the MPFR documentation's recipe for emulating IEEE formats).
 * The kernels are the C sources, compiled as C++ so that T can define comparison operators.
 *
 * Built on first use against MPFR (pfloat/_mp.py); PB_LIMBS (64-bit limbs per value) is set at
 * build time and bounds p <= 64 * PB_LIMBS.
 */
#include <mpfr.h>
#include <pthread.h>
#include <cmath>
#include <cstdint>
#include <cstring>

#ifndef PB_LIMBS
#define PB_LIMBS 2
#endif

struct T {
    int32_t kind;   /* 0 zero, 1 regular, 2 infinity, 3 NaN */
    int32_t sign;   /* 1 if negative */
    int64_t exp;    /* MPFR exponent: value = 0.d_1 d_2 ... * 2^exp */
    uint64_t limbs[PB_LIMBS];
};

static mpfr_prec_t g_p = 64;
static mpfr_exp_t g_mp_emin = -1000, g_mp_emax = 1000;
static int g_nlimbs = 1;
static pthread_mutex_t g_call_lock = PTHREAD_MUTEX_INITIALIZER;

static void set_range(void) {
    mpfr_set_emin(g_mp_emin);
    mpfr_set_emax(g_mp_emax);
}

static inline void view(mpfr_ptr v, const T &a) {
    int k = a.kind == 0 ? MPFR_ZERO_KIND : a.kind == 1 ? MPFR_REGULAR_KIND : a.kind == 2 ? MPFR_INF_KIND
                                                                                      : MPFR_NAN_KIND;
    if (a.sign && a.kind != 3) k = -k;
    mpfr_custom_init_set(v, k, a.kind == 1 ? (mpfr_exp_t)a.exp : 0, g_p, (void *)a.limbs);
}

static inline T pack(mpfr_srcptr r) {
    T t;
    std::memset(&t, 0, sizeof t);
    if (mpfr_nan_p(r)) { t.kind = 3; return t; }
    t.sign = mpfr_signbit(r) ? 1 : 0;
    if (mpfr_zero_p(r)) { t.kind = 0; return t; }
    if (mpfr_inf_p(r)) { t.kind = 2; return t; }
    t.kind = 1;
    t.exp = mpfr_get_exp(r);
    std::memcpy(t.limbs, mpfr_custom_get_significand(r), sizeof(uint64_t) * g_nlimbs);
    return t;
}

#define RESULT(r)                                   \
    uint64_t r##_buf[PB_LIMBS] = {0};               \
    mpfr_t r;                                       \
    mpfr_custom_init(r##_buf, g_p);                 \
    mpfr_custom_init_set(r, MPFR_ZERO_KIND, 0, g_p, r##_buf)

static inline T finish(mpfr_ptr r, int t) {
    t = mpfr_subnormalize(r, t, MPFR_RNDN);
    return pack(r);
}

static inline T m_add(const T &a, const T &b) {
    mpfr_t x, y; view(x, a); view(y, b); RESULT(r);
    return finish(r, mpfr_add(r, x, y, MPFR_RNDN));
}
static inline T m_sub(const T &a, const T &b) {
    mpfr_t x, y; view(x, a); view(y, b); RESULT(r);
    return finish(r, mpfr_sub(r, x, y, MPFR_RNDN));
}
static inline T m_mul(const T &a, const T &b) {
    mpfr_t x, y; view(x, a); view(y, b); RESULT(r);
    return finish(r, mpfr_mul(r, x, y, MPFR_RNDN));
}
static inline T m_div(const T &a, const T &b) {
    mpfr_t x, y; view(x, a); view(y, b); RESULT(r);
    return finish(r, mpfr_div(r, x, y, MPFR_RNDN));
}
static inline T m_sqrt(const T &a) {
    mpfr_t x; view(x, a); RESULT(r);
    return finish(r, mpfr_sqrt(r, x, MPFR_RNDN));
}
static inline T m_neg(const T &a) { T t = a; if (t.kind != 3) t.sign ^= 1; return t; }
static inline T m_abs(const T &a) { T t = a; t.sign = 0; return t; }
static inline T m_ldexp(const T &a, int k) {
    mpfr_t x; view(x, a); RESULT(r);
    return finish(r, mpfr_mul_2si(r, x, k, MPFR_RNDN));
}
static inline T m_rint(const T &a) {
    mpfr_t x; view(x, a); RESULT(r);
    return finish(r, mpfr_rint(r, x, MPFR_RNDN));
}
static inline T m_fromd(double v) {
    RESULT(r);
    return finish(r, mpfr_set_d(r, v, MPFR_RNDN));
}
static inline T m_i2t(long n) {
    RESULT(r);
    return finish(r, mpfr_set_si(r, n, MPFR_RNDN));
}
static inline double m_tod(const T &a) {
    mpfr_t x; view(x, a);
    return mpfr_get_d(x, MPFR_RNDN);
}

static inline bool operator<(const T &a, const T &b) { mpfr_t x, y; view(x, a); view(y, b); return mpfr_less_p(x, y); }
static inline bool operator>(const T &a, const T &b) { mpfr_t x, y; view(x, a); view(y, b); return mpfr_greater_p(x, y); }
static inline bool operator<=(const T &a, const T &b) { mpfr_t x, y; view(x, a); view(y, b); return mpfr_lessequal_p(x, y); }
static inline bool operator>=(const T &a, const T &b) { mpfr_t x, y; view(x, a); view(y, b); return mpfr_greaterequal_p(x, y); }
static inline bool operator==(const T &a, const T &b) { mpfr_t x, y; view(x, a); view(y, b); return mpfr_equal_p(x, y); }
static inline bool operator!=(const T &a, const T &b) { return !(a == b); }

static int m_setfmt(int p, int emin, int emax) {
    if (p < 2 || p > 64 * PB_LIMBS || emin > emax) return 1;
    g_p = p;
    g_nlimbs = (p + 63) / 64;
    g_mp_emin = (mpfr_exp_t)emin - p + 2; /* smallest subnormal 2^(emin-p+1) = 0.1b * 2^(emin-p+2) */
    g_mp_emax = (mpfr_exp_t)emax + 1;
    set_range();
    return 0;
}

#define ADD m_add
#define SUB m_sub
#define MUL m_mul
#define DIV m_div
#define SQRT m_sqrt
#define NEG m_neg
#define ABS m_abs
#define LDEXP m_ldexp
#define RINT m_rint
#define FROMD(x) m_fromd(x)
#define I2T(n) m_i2t(n)
#define TOD(a) m_tod(a)
#define SETFMT(p, emin, emax) m_setfmt((p), (emin), (emax))
#define IO T
#define LOADIO(x) (x)
#define STOREIO(x) (x)
#define THREAD_INIT() set_range()
#define ENTER() pthread_mutex_lock(&g_call_lock)
#define LEAVE() pthread_mutex_unlock(&g_call_lock)
#include "kernels.h"

/* ---- conversions and comparisons for the Python array type ---- */

/* binary64 values into the format (correctly rounded). */
PB_EXPORT int pb_from_double(const double *cst, size_t n, const double *x, T *out) {
    ENTER();
    int bad = m_setfmt((int)cst[0], (int)cst[1], (int)cst[2]);
    if (!bad)
        for (size_t i = 0; i < n; ++i) out[i] = m_fromd(x[i]);
    LEAVE();
    return bad;
}

/* Values of the format to binary64 (rounded to nearest); inexact counts the values that changed. */
PB_EXPORT int pb_to_double(const double *cst, size_t n, const T *a, double *out, long *inexact) {
    ENTER();
    int bad = m_setfmt((int)cst[0], (int)cst[1], (int)cst[2]);
    long count = 0;
    if (!bad) {
        mpfr_set_emin(mpfr_get_emin_min());
        mpfr_set_emax(mpfr_get_emax_max());
        for (size_t i = 0; i < n; ++i) {
            mpfr_t x;
            view(x, a[i]);
            out[i] = mpfr_get_d(x, MPFR_RNDN);
            if (!mpfr_nan_p(x) && mpfr_cmp_d(x, out[i]) != 0) ++count;
        }
        set_range();
    }
    *inexact = count;
    LEAVE();
    return bad;
}

/* Elementwise comparison: 0 <, 1 <=, 2 ==, 3 !=, 4 >, 5 >=. */
PB_EXPORT int pb_compare(const double *cst, int op, size_t n, const T *a, const T *b, int8_t *out) {
    ENTER();
    int bad = m_setfmt((int)cst[0], (int)cst[1], (int)cst[2]);
    if (!bad)
        for (size_t i = 0; i < n; ++i) {
            bool r = false;
            switch (op) {
                case 0: r = a[i] < b[i]; break;
                case 1: r = a[i] <= b[i]; break;
                case 2: r = a[i] == b[i]; break;
                case 3: r = a[i] != b[i]; break;
                case 4: r = a[i] > b[i]; break;
                case 5: r = a[i] >= b[i]; break;
            }
            out[i] = r;
        }
    LEAVE();
    return bad;
}

PB_EXPORT int pb_limbs(void) { return PB_LIMBS; }
