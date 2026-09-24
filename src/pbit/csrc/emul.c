/* pbit emulator: binary floating-point arithmetic with a p-bit significand.
 *
 * A format (p, emin, emax) has a p-bit significand (2 <= p <= 53), normal exponents
 * emin..emax, gradual underflow, infinities and NaNs (IEEE 754 semantics). Every +, -, *, /, sqrt
 * returns the correctly rounded result (round to nearest, ties to even); nothing is fused.
 *
 * How: the carrier is binary64. An operation computes the binary64 result s, then rounds s to
 * the format. Rounding s instead of the exact result t can only differ when s lies exactly on a
 * rounding midpoint of the format (the midpoints are binary64 numbers, and t is within half a
 * binary64 ulp of s). Only then is the side of t computed, exactly, with an error-free transform
 * (TwoSum for +, -; an FMA residual for *, /, sqrt, formed on scaled operands for tiny results
 * so that it cannot underflow). The FMA is used only there, to learn a sign; no model value is
 * ever fused. Exact binary64 is the hardware itself.
 *
 * Any other format needs emin >= -1022 + p + 1 so that all of its values, subnormals included,
 * are normal binary64 numbers (and emax <= 1023).
 */
#include <math.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static int g_shift = 0, g_emin = -1022, g_emax = 1023, g_native64 = 1;
/* fast-path parameters: low-bit mask and half-quantum at the normal-range rounding position,
   and the biased exponent range of the format's normal numbers */
static uint64_t g_mask = 0, g_half = 0;
static int g_exlo = 1, g_exhi = 2046;
/* event counters: per thread while computing, merged into the totals when a call or a worker ends */
static _Thread_local long g_nonfinite = 0, g_notfmt = 0, g_overflow = 0, g_underflow = 0;
static long g_totals[4] = {0, 0, 0, 0};
static pthread_mutex_t g_totals_lock = PTHREAD_MUTEX_INITIALIZER;
/* the format lives in process globals, so calls into the emulator are serialized */
static pthread_mutex_t g_call_lock = PTHREAD_MUTEX_INITIALIZER;

static void flush_counters(void) {
    pthread_mutex_lock(&g_totals_lock);
    g_totals[0] += g_nonfinite; g_totals[1] += g_notfmt; g_totals[2] += g_overflow; g_totals[3] += g_underflow;
    pthread_mutex_unlock(&g_totals_lock);
    g_nonfinite = g_notfmt = g_overflow = g_underflow = 0;
}

#define SIGNBIT 0x8000000000000000ull
#define FRACMASK 0x000fffffffffffffull
#define TINY 0x1p-900

static inline uint64_t to_bits(double x) { uint64_t u; memcpy(&u, &x, 8); return u; }
static inline double from_bits(uint64_t u) { double x; memcpy(&x, &u, 8); return x; }
static inline int sgn(double e) { return (e > 0) - (e < 0); }

/* The rounding position of |s| (in binary64 bits), or 0 when s needs no rounding. */
static inline int shift_of(uint64_t mag) {
    int e = (int)(mag >> 52) - 1023;
    return g_shift + (e < g_emin ? g_emin - e : 0);
}

/* 1 if s sits exactly on a rounding midpoint of the format. */
static inline int at_midpoint(double s) {
    if (g_native64) return 0;
    uint64_t mag = to_bits(s) & ~SIGNBIT;
    int ex = (int)(mag >> 52);
    if (ex == 0 || ex == 0x7ff) return 0;
    int shift = shift_of(mag);
    if (shift == 0) return 0;
    if (shift <= 52) {
        uint64_t one = 1ull << shift;
        return (mag & (one - 1)) == (one >> 1);
    }
    if (shift == 53) return (mag & FRACMASK) == 0;
    return 0;
}

/* Round binary64 s to the format; dir is the sign of (exact - s) (only read at a midpoint). */
static inline double rnd(double s, int dir) {
    uint64_t u = to_bits(s);
    int ex = (int)((u >> 52) & 0x7ff);
    if (ex == 0x7ff) { ++g_nonfinite; return s; }
    if (s == 0.0) return s;
    if (g_native64) { if (ex == 0) ++g_underflow; return s; }
    uint64_t sign = u & SIGNBIT, mag = u ^ sign;
    if (ex == 0) { ++g_underflow; return from_bits(sign); }
    int shift = shift_of(mag);
    int mdir = sign ? -dir : dir;
    if (shift == 0) {
        /* already a format value */
    } else if (shift <= 52) {
        uint64_t one = 1ull << shift, half = one >> 1, low = mag & (one - 1);
        mag -= low;
        /* parity of the kept significand; at shift 52 only the implicit leading 1 is kept */
        int odd = shift == 52 ? 1 : (mag & one) != 0;
        if (low > half || (low == half && (mdir > 0 || (mdir == 0 && odd)))) mag += one;
    } else if (shift == 53) {
        /* |s| in [q/2, q), q = 2^(e+1): above the midpoint rounds to q; at it, by the residual,
           else to the even neighbour 0 */
        mag = ((mag & FRACMASK) != 0 || mdir > 0) ? (uint64_t)(ex + 1) << 52 : 0;
    } else {
        mag = 0;
    }
    if (mag == 0) { ++g_underflow; return from_bits(sign); }
    int e2 = (int)(mag >> 52) - 1023;
    if (e2 > g_emax) { ++g_overflow; return copysign(INFINITY, s); }
    if (e2 < g_emin) ++g_underflow;
    return from_bits(sign | mag);
}

#define HOT static inline __attribute__((always_inline))
#define COLD static __attribute__((noinline))

/* The common case in one integer add and mask: s is zero, or normal in the format, not on a
 * midpoint, and rounding does not overflow. Returns 1 with *out set, or 0 for the general path. */
HOT int round_fast(double s, double *out) {
    uint64_t u = to_bits(s), sign = u & SIGNBIT, mag = u ^ sign;
    if (mag == 0) { *out = s; return 1; }
    int ex = (int)(mag >> 52);
    if (ex < g_exlo || ex > g_exhi) return 0;
    uint64_t low = mag & g_mask;
    if (g_mask && low == g_half) return 0;
    uint64_t r = (mag + g_half) & ~g_mask;
    if ((int)(r >> 52) > g_exhi) return 0;
    *out = from_bits(sign | r);
    return 1;
}

/* General paths: the residual sign is computed only when s sits on a midpoint. */
COLD double add_slow(double a, double b, double s) {
    int d = 0;
    if (at_midpoint(s)) {
        double bb = s - a, e = (a - (s - bb)) + (b - bb); /* TwoSum: a + b = s + e exactly */
        d = sgn(e);
    }
    return rnd(s, d);
}
COLD double mul_slow(double a, double b, double s) {
    int d = 0;
    if (at_midpoint(s)) {
        double r;
        if (fabs(s) < TINY) r = fabs(a) < fabs(b) ? fma(ldexp(a, 256), b, -ldexp(s, 256))
                                                 : fma(a, ldexp(b, 256), -ldexp(s, 256));
        else r = fma(a, b, -s); /* a b - s, exact */
        d = sgn(r);
    }
    return rnd(s, d);
}
COLD double div_slow(double a, double b, double q) {
    int d = 0;
    if (at_midpoint(q)) {
        double r = fabs(a) < TINY ? fma(-ldexp(q, 256), b, ldexp(a, 256)) : fma(-q, b, a); /* a - q b */
        d = b < 0 ? -sgn(r) : sgn(r);
    }
    return rnd(q, d);
}
COLD double sqrt_slow(double a, double s) {
    int d = 0;
    if (at_midpoint(s)) {
        double r = a < TINY ? fma(-ldexp(s, 256), ldexp(s, 256), ldexp(a, 512)) : fma(-s, s, a); /* a - s^2 */
        d = sgn(r);
    }
    return rnd(s, d);
}

HOT double e_add(double a, double b) {
    double s = a + b, r;
    return round_fast(s, &r) ? r : add_slow(a, b, s);
}
HOT double e_sub(double a, double b) { return e_add(a, -b); }
HOT double e_mul(double a, double b) {
    double s = a * b, r;
    return round_fast(s, &r) ? r : mul_slow(a, b, s);
}
HOT double e_div(double a, double b) {
    double q = a / b, r;
    return round_fast(q, &r) ? r : div_slow(a, b, q);
}
HOT double e_sqrt(double a) {
    double s = sqrt(a), r;
    return round_fast(s, &r) ? r : sqrt_slow(a, s);
}
static inline double e_fromd(double x) {
    if (x == x && rnd(x, 0) != x) ++g_notfmt;
    return x;
}
static inline double e_i2t(long n) { return rnd((double)n, 0); }
static inline int e_setfmt(int p, int emin, int emax) {
    if (p < 2 || p > 53 || emax > 1023 || emin > emax) return 1;
    g_native64 = (p == 53 && emin == -1022 && emax == 1023);
    if (!g_native64 && emin < -1022 + p + 1) return 1;
    g_shift = 53 - p; g_emin = emin; g_emax = emax;
    g_mask = g_shift ? (1ull << g_shift) - 1 : 0;
    g_half = g_shift ? 1ull << (g_shift - 1) : 0;
    g_exlo = emin + 1023; g_exhi = emax + 1023;
    return 0;
}

#define T double
#define ADD e_add
#define SUB e_sub
#define MUL e_mul
#define DIV e_div
#define SQRT e_sqrt
#define NEG(a) (-(a))
#define ABS(a) fabs(a)
#define LDEXP(a, k) rnd(ldexp((a), (k)), 0)
#define RINT(a) nearbyint(a)
#define FROMD(x) e_fromd(x)
#define I2T(n) e_i2t(n)
#define TOD(a) (a)
#define SETFMT(p, emin, emax) e_setfmt((p), (emin), (emax))
#define THREAD_EXIT() flush_counters()
#define ENTER() pthread_mutex_lock(&g_call_lock)
#define LEAVE() (flush_counters(), pthread_mutex_unlock(&g_call_lock))
#include "kernels.h"

/* Round binary64 values into the format (nearest, ties to even). */
int pb_round(const double *cst, size_t n, const double *x, double *out) {
    ENTER();
    int bad = e_setfmt((int)cst[0], (int)cst[1], (int)cst[2]);
    if (!bad)
        for (size_t i = 0; i < n; ++i) out[i] = rnd(x[i], 0);
    LEAVE();
    return bad;
}

/* [0] non-finite results, [1] inputs that were not format values, [2] overflows, [3] results in
   the subnormal range or flushed to zero. Reset on read. */
void pb_events(long *out) {
    flush_counters();
    pthread_mutex_lock(&g_totals_lock);
    for (int i = 0; i < 4; ++i) { out[i] = g_totals[i]; g_totals[i] = 0; }
    pthread_mutex_unlock(&g_totals_lock);
}
