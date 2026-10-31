#!/usr/bin/env python3
"""Exact algebraic certificates for N32_converse_theorem.tex.

Requires Python 3.9+ and only its standard library. Run:
    python3 verify_N32.py

No floating-point arithmetic is used. The analytic arguments are proved
in the accompanying manuscript; this script checks their matrix input.
The level 36 files, if present in the parent directory, are only read.
"""

from fractions import Fraction as Q
from hashlib import sha256
from itertools import product
from math import gcd
from pathlib import Path


def mat(*rows):
    return tuple(tuple(Q(x) for x in row) for row in rows)


def eye(n):
    return tuple(tuple(Q(i == j) for j in range(n)) for i in range(n))


def mul(a, b):
    return tuple(tuple(sum(a[i][r] * b[r][j] for r in range(len(b)))
                       for j in range(len(b[0]))) for i in range(len(a)))


def scale(c, a):
    return tuple(tuple(c * x for x in row) for row in a)


def add(a, b):
    return tuple(tuple(x + y for x, y in zip(ra, rb))
                 for ra, rb in zip(a, b))


def det(a):
    if len(a) == 1:
        return a[0][0]
    return sum((-1) ** j * a[0][j] * det(tuple(
        row[:j] + row[j+1:] for row in a[1:])) for j in range(len(a)))


def trace(a):
    return sum(a[i][i] for i in range(len(a)))


def inv(a):
    assert len(a) == 2 and det(a) != 0
    x, y = a[0]
    z, w = a[1]
    return scale(1 / det(a), mat((w, -y), (-z, x)))


def power(a, n):
    if n < 0:
        return power(inv(a), -n)
    result = eye(len(a))
    for _ in range(n):
        result = mul(result, a)
    return result


def beta(t):
    return mat((1, t), (0, 1))


def W(n):
    return mat((1, 0), (n, 1))


def H(n):
    return mat((0, -1), (n, 0))


def chi(a):
    assert det(a) == 1 and all(x.denominator == 1 for row in a for x in row)
    assert int(a[1][0]) % 8 == 0
    return (-1) ** ((int(a[0][1]) + int(a[1][0]) // 8) % 2)


I = eye(2)
T = beta(1)
D = mat((1, 0), (0, 2))


def verify_parameter_module():
    # Each entry of A^2-I and (AB)^3-I is a polynomial in tau of
    # degree <= 3. The characteristic identity has degree <= 3 in
    # each of tau and X. Four exact values in each variable therefore
    # certify the polynomial identities for all complex parameters.
    values = (-1, 0, 1, 2)
    for eps in (1, -1):
        a = mat((eps, 0, 0), (0, 0, 1), (0, 1, 0))
        for tau in values:
            b = mat((0, 1, 0), (-1, tau, 0), (tau, 0, -eps))
            assert power(a, 2) == eye(3)
            assert power(mul(a, b), 3) == eye(3)
            for x in values:
                lhs = det(add(scale(x, eye(3)), scale(-1, b)))
                rhs = (x + eps) * (x*x - tau*x + 1)
                assert lhs == rhs
    print('PASS: parameterized module identities (exact polynomial certificate).')


def verify_uniform_relation():
    expected_r = {1: -2, 2: 2, 4: 0, 8: 1}
    for m in (1, 2, 4, 8):
        n = 4*m
        sigma = 1 if m % 3 == 2 else -1
        a = mat((3, 2), (2*sigma*m, Q(1+4*sigma*m, 3)))
        b = mat((3, -2), (-2*sigma*m, Q(1+4*sigma*m, 3)))
        c = mat((-3, -2), (sigma*m, Q(2*sigma*m-1, 3)))
        cp = mat((-3, 2), (-sigma*m, Q(2*sigma*m-1, 3)))
        assert all(det(x) == 1 for x in (a, b, c, cp))
        assert mul(inv(D), mul(H(n), D)) == scale(2, H(m))
        assert mul(H(m), mul(inv(T), inv(H(m)))) == W(m)
        assert mul(power(W(m), -sigma), a) == scale(-1, c)
        assert mul(power(W(m), sigma), b) == scale(-1, cp)
        r = Q(2, 3) + Q(8, 3*sigma*m)
        assert r == expected_r[m]
        assert cp == mul(beta(r), mul(inv(c), beta(-r)))
        e = mul(inv(c), beta(Q(4, 3)-r))
        assert det(e) == 1 and trace(e) == -Q(2, 3)
        assert -2 < trace(e) < 2 and trace(e).denominator != 1
        for residue, chosen_c, expected in ((-1, -sigma, a), (1, sigma, b)):
            assert (1+n*residue*chosen_c) % 3 == 0
            gamma = mat((3, -residue),
                        (-n*chosen_c, Q(1+n*residue*chosen_c, 3)))
            assert det(gamma) == 1
            assert mul(inv(D), mul(gamma, D)) == expected
            assert mul(gamma, beta(Q(residue, 3))) == mat(
                (3, 0), (-n*chosen_c, Q(1, 3)))
    print('PASS: uniform prime relation at N=4,8,16,32; elliptic trace -2/3.')


def verify_level32():
    c = mat((-3, -2), (8, 5))
    cp = mat((-3, 2), (-8, 5))
    e = mat((5, Q(11, 3)), (-8, -Q(17, 3)))
    assert cp == mul(T, mul(inv(c), inv(T)))
    assert e == mul(inv(c), beta(Q(1, 3)))
    assert det(e) == 1 and trace(e) == -Q(2, 3)
    # The base generators and Schreier generators in the generation proof.
    assert mul(D, mul(power(T, 2), inv(D))) == T
    assert mul(D, mul(W(2), inv(D))) == W(4)
    assert power(W(4), 2) == W(8)
    assert c == power(mul(inv(W(4)), T), 2)
    assert mul(W(4), mul(T, inv(W(4)))) == mul(W(8), mul(c, inv(T)))
    assert all(chi(x) == -1 for x in (T, W(8), c))
    assert chi(scale(-1, I)) == 1
    # Independent finite verification of the character on its full
    # modulus: chi depends only on a matrix modulo 16, with c=0 mod 8.
    residues = []
    for a, b, lower_c in product(range(1, 16, 2), range(16), (0, 8)):
        d = pow(a, -1, 16) * (1+b*lower_c) % 16
        residues.append((a, b, lower_c, d))

    def modmul(x, y):
        a, b, c0, d = x
        A, B, C, D0 = y
        return ((a*A+b*C) % 16, (a*B+b*D0) % 16,
                (c0*A+d*C) % 16, (c0*B+d*D0) % 16)

    def parity(x):
        return (x[1] + x[2]//8) % 2

    assert len(residues) == 256
    for x, y in product(residues, repeat=2):
        assert parity(modmul(x, y)) == (parity(x)+parity(y)) % 2
    print('PASS: N=32 matrices, Schreier identities, and all 65,536 character products.')


def verify_cusps():
    # Left cosets of Gamma_0(8) are primitive bottom rows modulo 8,
    # up to multiplication by a unit. Right T-orbits are the cusps.
    units = (1, 3, 5, 7)

    def normalize(pair):
        return min(tuple(u*x % 8 for x in pair) for u in units)

    def translate(pair):
        c, d = pair
        return normalize((c, c+d))

    points = {normalize((c, d)) for c, d in product(range(8), repeat=2)
              if gcd(gcd(c, d), 8) == 1}
    assert len(points) == 12
    orbits = []
    remaining = set(points)
    while remaining:
        start = min(remaining)
        orbit = {start}
        point = translate(start)
        while point != start:
            assert point not in orbit
            orbit.add(point)
            point = translate(point)
        remaining -= orbit
        orbits.append(orbit)
    assert sorted(map(len, orbits)) == [1, 1, 2, 8]
    representatives = [('infinity', I, 1), ('0', H(1), 8),
                       ('1/2', W(2), 2), ('1/4', W(4), 1)]
    used = set()
    for name, alpha, width in representatives:
        point = normalize(tuple(int(x) for x in alpha[1]))
        index = next(i for i, orbit in enumerate(orbits) if point in orbit)
        assert index not in used and len(orbits[index]) == width
        used.add(index)
        stabilizer = mul(alpha, mul(power(T, width), inv(alpha)))
        assert chi(stabilizer) == -1
    assert len(used) == 4
    print('PASS: all four cusp orbits, widths 1,8,2,1, and negative stabilizer signs.')


def verify_preserved_files():
    original_dir = Path(__file__).resolve().parent.parent
    expected = {
        'N36_converse_theorem.tex':
            'e57da53308b60558ff0c8fd3c0c805fffe86169917c3069f4cdc8129f6fa582f',
        'N36_converse_theorem.pdf':
            'b6889ba06343f305de08ae234222f782940650c94dcf86f8ca100c4f06b0ebbd',
        'verify_N36.py':
            'b37d1ac7d22ac21e52a0c6c046264b508b288102f86c17cc062ae1baa347315a',
    }
    checked = 0
    for filename, digest in expected.items():
        path = original_dir / filename
        if path.exists():
            assert sha256(path.read_bytes()).hexdigest() == digest, filename
            checked += 1
    print(f'PASS: {checked} original level 36 files checked unchanged.'
          if checked else 'INFO: original level 36 files are not in the parent directory; skipped.')


def main():
    verify_parameter_module()
    verify_uniform_relation()
    verify_level32()
    verify_cusps()
    verify_preserved_files()
    print('All exact verification checks passed.')


if __name__ == '__main__':
    main()
