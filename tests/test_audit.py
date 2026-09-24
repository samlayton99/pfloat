"""The static no-leakage audit, and negative controls showing it catches planted leaks."""
import shutil

import pytest

from pbit import audit

pytestmark = pytest.mark.skipif(shutil.which("clang") is None, reason="the audit parses the C with clang")


def test_only_the_documented_exception():
    assert audit.audit() == audit.ALLOWED


@pytest.mark.parametrize("old,new,expect", [
    ("T dtemp = ADD(MUL(c, x[ix]), MUL(s, y[iy]));", "T dtemp = c * x[ix] + MUL(s, y[iy]);", "floating '*'"),
    ("T d = SQRT(ADD(MUL(f, f), MUL(g, g)));", "T d = sqrt(ADD(MUL(f, f), MUL(g, g)));", "call sqrt"),
    ("sminoa = DIV(sminoa, SQRT(I2T(n)));", "sminoa = DIV(sminoa, SQRT((double)n));", "cast IntegralToFloating"),
    ("T q = DIV(z, w);", "T q = z / w;", "floating '/'"),
    ("", "", None),  # control: an unchanged copy is clean
])
def test_catches_planted_leaks(tmp_path, old, new, expect):
    for name in ("emul.c", "kernels.h", "lapack_gelss.h"):
        (tmp_path / name).write_text((audit.CSRC / name).read_text())
    path = tmp_path / "lapack_gelss.h"
    if expect is not None:
        text = path.read_text()
        assert old in text
        path.write_text(text.replace(old, new, 1))
    extra = [f for f in audit.audit(tmp_path) if f not in audit.ALLOWED]
    if expect is None:
        assert extra == []
    else:
        assert any(expect in f["what"] for f in extra), extra
