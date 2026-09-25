/* The pfloat kernels on native hardware: -DPB_TYPE=1 binary64 (double), -DPB_TYPE=2 binary32
 * (float). Built without fused multiply-add or auto-vectorization; each operation is a
 * separate function returning T, so every result is rounded to T. Bit-identical to the emulator
 * in the same format (checked in the tests); used as a fast path for exactly those formats.
 */
#include <math.h>
#include <stddef.h>

#if PB_TYPE == 1
typedef double T;
#define NATIVE_P 53
#define NATIVE_EMIN -1022
#define NATIVE_EMAX 1023
static inline T n_sqrt(T a) { return sqrt(a); }
static inline T n_rint(T a) { return nearbyint(a); }
static inline T n_ldexp(T a, int k) { return ldexp(a, k); }
static inline T n_abs(T a) { return fabs(a); }
#elif PB_TYPE == 2
typedef float T;
#define NATIVE_P 24
#define NATIVE_EMIN -126
#define NATIVE_EMAX 127
static inline T n_sqrt(T a) { return sqrtf(a); }
static inline T n_rint(T a) { return nearbyintf(a); }
static inline T n_ldexp(T a, int k) { return ldexpf(a, k); }
static inline T n_abs(T a) { return fabsf(a); }
#else
#error "PB_TYPE must be 1 (double) or 2 (float)"
#endif

static inline T n_add(T a, T b) { T r = a + b; return r; }
static inline T n_sub(T a, T b) { T r = a - b; return r; }
static inline T n_mul(T a, T b) { T r = a * b; return r; }
static inline T n_div(T a, T b) { T r = a / b; return r; }
static inline T n_neg(T a) { T r = -a; return r; }
static inline T n_fromd(double x) { return (T)x; } /* exact: inputs are already values of T */
static inline T n_i2t(long n) { return (T)n; }
static inline int n_setfmt(int p, int emin, int emax) {
    return p != NATIVE_P || emin != NATIVE_EMIN || emax != NATIVE_EMAX;
}

#define ADD n_add
#define SUB n_sub
#define MUL n_mul
#define DIV n_div
#define SQRT n_sqrt
#define NEG n_neg
#define ABS n_abs
#define LDEXP n_ldexp
#define RINT n_rint
#define FROMD(x) n_fromd(x)
#define I2T(n) n_i2t(n)
#define TOD(a) ((double)(a))
#define SETFMT(p, emin, emax) n_setfmt((p), (emin), (emax))
#define PB_PAR_MIN_WORK (1L << 20) /* native operations are ~10x cheaper than emulated ones */
#include "kernels.h"
