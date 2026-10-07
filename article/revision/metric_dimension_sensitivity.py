"""How large is the evaluation-dimension effect, as a function of how much
of a method's velocity norm lies outside the signal subspace?

This is the quantity that decides whether the benchmark comparison is
compromised or merely imperfect. Reported honestly: ICCoh in the signal
subspace is the accuracy we actually care about; ICCoh measured in d
ambient dimensions is what the benchmark reports.
"""
import numpy as np
from scipy.spatial import cKDTree

rng = np.random.default_rng(0)
N, D_MAX = 2500, 2000

n_stem = n_br = N // 3
t0 = rng.uniform(0, 1, n_stem)
t1 = rng.uniform(0, 1, n_br)
Z2 = np.vstack([np.c_[t0, np.zeros(n_stem)],
                np.c_[1 + t1, 0.6 * t1], np.c_[1 + t1, -0.6 * t1]])
U2 = np.vstack([np.tile([1, 0], (n_stem, 1)),
                np.tile([1, .6], (n_br, 1)), np.tile([1, -.6], (n_br, 1))])
U2 = U2 / np.linalg.norm(U2, axis=1, keepdims=True)
labels = np.r_[np.zeros(n_stem), np.ones(n_br), 2 * np.ones(n_br)].astype(int)
n = len(Z2)

X = np.zeros((n, D_MAX)); X[:, :2] = Z2
X[:, 2:] = rng.normal(0, .05, (n, D_MAX - 2))


def velocity(ang, nf, seed):
    r = np.random.default_rng(seed)
    th = np.arctan2(U2[:, 1], U2[:, 0]) + r.normal(0, ang, n)
    V = np.zeros((n, D_MAX)); V[:, 0], V[:, 1] = np.cos(th), np.sin(th)
    nz = r.normal(0, 1, (n, D_MAX - 2))
    nz /= np.linalg.norm(nz, axis=1, keepdims=True)
    V[:, 2:] = nz * nf
    return V


def iccoh(V, nbr):
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    o = []
    for i in range(len(V)):
        s = nbr[i][labels[nbr[i]] == labels[i]]
        if len(s): o.append(float((Vn[i] @ Vn[s].T).mean()))
    return float(np.mean(o))


NBR = {}
for d in (2, 10, 50, 2000):
    tree = cKDTree(X[:, :d]); _, idx = tree.query(X[:, :d], k=31)
    NBR[d] = idx[:, 1:]

print("Method accuracy is IDENTICAL across each row (angular error 0.30 rad).")
print("Only the nuisance fraction of the velocity norm, and the evaluation")
print("dimension, change.\n")
print(f"{'nuisance frac':>13} | {'ICCoh @2D':>9} | {'@10D':>7} | {'@50D':>7} "
      f"| {'@2000D':>7} | {'2D - 2000D':>10}")
print("-" * 74)
for nf in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
    V = velocity(0.30, nf, 7)
    vals = {d: iccoh(V[:, :d], NBR[d]) for d in (2, 10, 50, 2000)}
    print(f"{nf:>13.2f} | {vals[2]:>9.4f} | {vals[10]:>7.4f} | "
          f"{vals[50]:>7.4f} | {vals[2000]:>7.4f} | "
          f"{vals[2]-vals[2000]:>+10.4f}")

print("\n\nWhen does the ranking actually flip?")
print("Method A: angular error 0.30 (better). Method B: 0.80 (worse).")
print("A is evaluated at 2000-D, B at 10-D - the benchmark's actual setup.\n")
print(f"{'A nuisance frac':>15} | {'A @2000D':>9} | {'B @10D':>8} | verdict")
print("-" * 62)
for nf_a in (0.35, 1.0, 2.0, 3.0, 4.0, 6.0):
    VA = velocity(0.30, nf_a, 11)
    VB = velocity(0.80, 0.35, 12)
    a = iccoh(VA[:, :2000], NBR[2000]); b = iccoh(VB[:, :10], NBR[10])
    verdict = "RANKING FLIPPED" if b > a else "order preserved"
    print(f"{nf_a:>15.2f} | {a:>9.4f} | {b:>8.4f} | {verdict}")

print("\n\nNull scale: what an ICCoh of 0.5 means in each space.")
print("(random unit vectors; sd of pairwise cosine)\n")
print(f"{'dim':>6} | {'sd of null':>10} | {'0.5 is this many sd from random':>32}")
print("-" * 56)
for d in (2, 10, 30, 50, 100, 2000):
    r = rng.normal(0, 1, (4000, d))
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    c = (r[:2000] * r[2000:]).sum(1)
    print(f"{d:>6} | {c.std():>10.4f} | {0.5/c.std():>32.1f}")
