#!/usr/bin/env python3
"""Exact algebraic checks for N36_converse_theorem.tex; standard library only.

This script checks the matrix algebra, not the analytic arguments.
Run with: python3 verify_N36.py
"""

from collections import deque
from fractions import Fraction as Q


def matrix(rows):
    return tuple(tuple(Q(x) for x in row) for row in rows)


def eye(n):
    return matrix([[int(i == j) for j in range(n)] for i in range(n)])


def mul(a, b):
    return tuple(
        tuple(sum(a[i][r] * b[r][j] for r in range(len(b)))
              for j in range(len(b[0])))
        for i in range(len(a))
    )


def scale(c, a):
    return tuple(tuple(c * x for x in row) for row in a)


def add(a, b):
    return tuple(tuple(x + y for x, y in zip(r, s)) for r, s in zip(a, b))


def inverse(a):
    n = len(a)
    rows = [list(a[i]) + list(eye(n)[i]) for i in range(n)]
    for j in range(n):
        pivot = next(i for i in range(j, n) if rows[i][j])
        rows[j], rows[pivot] = rows[pivot], rows[j]
        c = rows[j][j]
        rows[j] = [x / c for x in rows[j]]
        for i in range(n):
            if i != j:
                c = rows[i][j]
                rows[i] = [x - c * y for x, y in zip(rows[i], rows[j])]
    return tuple(tuple(row[n:]) for row in rows)


def power(a, n):
    if n < 0:
        return power(inverse(a), -n)
    b = eye(len(a))
    while n:
        if n & 1:
            b = mul(b, a)
        a = mul(a, a)
        n //= 2
    return b


def product(*args):
    b = eye(len(args[0]))
    for a in args:
        b = mul(b, a)
    return b


def beta(t):
    return matrix([[1, t], [0, 1]])


def det2(a):
    return a[0][0] * a[1][1] - a[0][1] * a[1][0]


S = matrix([[0, -1], [1, 0]])
T = beta(1)
D = matrix([[1, 0], [0, 6]])
H = matrix([[0, -1], [36, 0]])
W = matrix([[1, 0], [36, 1]])

assert product(S, T, S) == product(inverse(T), S, inverse(T))
assert product(H, inverse(T), inverse(H)) == W

gammas = {
    1: matrix([[5, -1], [36, -7]]),
    -1: matrix([[5, 1], [-36, -7]]),
    2: matrix([[5, -2], [-72, 29]]),
    -2: matrix([[5, 2], [72, 29]]),
}
deltas = {b: product(inverse(D), a, D) for b, a in gammas.items()}
cs = {1: -1, -1: 1, 2: 2, -2: -2}
assert {c % 5 for c in cs.values()} == {1, 2, 3, 4}
for b, gamma in gammas.items():
    c = cs[b]
    assert det2(gamma) == 1
    assert (36 * b * c + 1) % 5 == 0
    assert scale(5, mul(gamma, beta(Q(b, 5)))) == matrix(
        [[25, 0], [-180 * c, 1]]
    )
    assert product(inverse(D), beta(Q(b, 5)), D) == beta(Q(6 * b, 5))
    if abs(b) == 1:
        word = product(power(T, b), S, power(T, 6 * b), S, power(T, -b))
    else:
        word = power(product(S, power(T, b), S, power(T, -b)), 2)
    assert word == deltas[b]

for r in range(5):
    assert product(H, matrix([[1, r], [0, 5]]), inverse(H)) == matrix(
        [[5, 0], [-36 * r, 1]]
    )
assert product(H, matrix([[5, 0], [0, 1]]), inverse(H)) == matrix(
    [[1, 0], [0, 5]]
)
for c in range(-2, 3):
    assert mul(inverse(W), matrix([[25, 0], [-180 * c, 1]])) == matrix(
        [[25, 0], [-180 * (c + 5), 1]]
    )
print("Prime relation, all four conjugations, and exact word identities: OK")

representations = {}
for epsilon in (1, -1):
    A = matrix([[epsilon, 0, 0], [0, 0, 1], [0, 1, 0]])
    B = matrix([[0, 1, 0], [-1, 1, 0], [1, 0, -epsilon]])
    I = eye(3)
    assert power(A, 2) == power(mul(A, B), 3) == power(B, 6) == I
    h = matrix([[1, 0, 0]])
    assert mul(h, add(add(power(B, 2), scale(-1, B)), I)) == matrix([[0, 0, 0]])
    if epsilon == 1:
        assert power(B, 3) == scale(-1, I)
        representations["positive Fricke sign"] = (A, B)
    else:
        u = matrix([[0, 1, 1]])
        assert mul(u, A) == mul(u, B) == u
        for b in (1, -1, 2, -2):
            if abs(b) == 1:
                R = product(power(B, b), A, power(B, 6 * b), A, power(B, -b))
                expected = h
            else:
                R = power(product(A, power(B, b), A, power(B, -b)), 2)
                expected = add(h, scale(-b, u))
            assert mul(h, R) == expected
        print("Negative sign: u is invariant, and the four defects are 0,0,-2u,2u: OK")
        A2 = scale(-1, eye(2))
        B2 = matrix([[0, 1], [-1, 1]])
        assert power(A2, 2) == power(mul(A2, B2), 3) == eye(2)
        assert power(B2, 3) == scale(-1, eye(2))
        representations["negative sign after u=0"] = (A2, B2)
print("Full modular-group representation relations: OK")


def residue_key(a):
    flat = tuple(int(x) % 6 for row in a for x in row)
    return min(flat, tuple((-x) % 6 for x in flat))


def check_mod6(A, B):
    """Check every edge, not just a selection of words or generators."""
    identity = eye(len(A))
    seen = {residue_key(eye(2)): identity}
    queue = deque([(eye(2), identity)])
    edges = 0
    while queue:
        m, rho = queue.popleft()
        for generator, image in ((S, A), (T, B)):
            n = mul(m, generator)
            r = mul(rho, image)
            key = residue_key(n)
            edges += 1
            if key in seen:
                assert seen[key] == r
            else:
                seen[key] = r
                queue.append((matrix([key[:2], key[2:]]), r))
    assert len(seen) == 72 and edges == 144
    assert seen[residue_key(eye(2))] == identity
    return len(set(seen.values()))


for label, pair in representations.items():
    image_size = check_mod6(*pair)
    print(f"{label}: all 72 projective residue classes, 144 edges checked; image size {image_size}")

E = mul(S, beta(Q(1, 5)))
assert det2(E) == 1 and E[0][0] + E[1][1] == Q(1, 5)
assert product(beta(Q(24, 5)), power(T, -5)) == beta(Q(-1, 5))
print("Final elliptic matrix: determinant 1, trace 1/5; period reduction: OK")
print("All exact algebraic checks passed.")
