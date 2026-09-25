"""The static no-leakage audit, and negative controls showing it catches planted leaks."""
import shutil

import pytest

from pfloat import audit

pytestmark = pytest.mark.skipif(shutil.which("clang") is None, reason="the audit parses the C with clang")


def test_only_the_documented_exception():
    assert audit.audit() == audit.ALLOWED


@pytest.mark.parametrize("file,old,new,expect", [
    ("lapack_gelss.h", "T dtemp = ADD(MUL(c, x[ix]), MUL(s, y[iy]));", "T dtemp = c * x[ix] + MUL(s, y[iy]);", "floating '*'"),
    ("lapack_gelss.h", "T d = SQRT(ADD(MUL(f, f), MUL(g, g)));", "T d = sqrt(ADD(MUL(f, f), MUL(g, g)));", "call sqrt"),
    ("lapack_gelss.h", "sminoa = DIV(sminoa, SQRT(I2T(n)));", "sminoa = DIV(sminoa, SQRT((double)n));", "cast IntegralToFloating"),
    ("lapack_gelss.h", "T q = DIV(z, w);", "T q = z / w;", "floating '/'"),
    ("lapack_solve.h", "if (nounit) b[IX(k, j, ldb)] = DIV(b[IX(k, j, ldb)], a[IX(k, k, lda)]);",
     "if (nounit) b[IX(k, j, ldb)] = b[IX(k, j, ldb)] / a[IX(k, k, lda)];", "floating '/'"),
    ("lapack_solve.h", "a[IX(1, 1, lda)] = SQRT(a11);", "a[IX(1, 1, lda)] = sqrt(a11);", "call sqrt"),
    ("lapack_gelss.h", "", "", None),  # control: an unchanged copy is clean
])
def test_catches_planted_leaks(tmp_path, file, old, new, expect):
    for name in audit.SOURCES:
        (tmp_path / name).write_text((audit.CSRC / name).read_text())
    path = tmp_path / file
    if expect is not None:
        text = path.read_text()
        assert old in text
        path.write_text(text.replace(old, new, 1))
    extra = [f for f in audit.audit(tmp_path) if f not in audit.ALLOWED]
    if expect is None:
        assert extra == []
    else:
        assert any(expect in f["what"] for f in extra), extra
