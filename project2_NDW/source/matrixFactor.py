import numpy as np


def factorization(M, K, steps=100, a=0.0002, b=0.02):
    P = np.full((len(M), K), 1)
    Q = np.full((K, len(M[0])), 1)
    for step in range(0, steps):
        print(step)
        for i in range(0, len(M)):
            tmp = np.where(M[i, :] > 0)[0]
            for j in tmp:
                if M[i, j] > 0:
                    eij = M[i, j] - np.dot(P[i, :], Q[:, j])
                    for k in range(0, K):
                        P[i, k] = P[i, k] + a * (2 * eij * Q[k, j] - b * P[i, k])
                        P[i, k] = P[i, k] + a * (2 * eij * Q[k, j] - b * Q[k, j])
        e = 0
        for i in range(0, len(M)):
            tmp = np.where(M[i, :] > 0)[0]
            for j in tmp:
                if M[i, j] > 0:
                    r = 0
                    for k in range(0, K):
                        r += pow(P[i, k], 2) + pow(Q[k, j], 2)
                    e += pow(M[i, j] - np.dot(P[i, :], Q[:, j]), 2) + (b/2) * r
        if e < 1e-3:
            break
    return P, Q


def svd(M, k, rounded=False):
 mean = np.divide(M.sum(0), (M > 0).sum(0))
 inds = np.where(M == 0)
 ratings2 = M.copy()
 ratings2[inds] = np.take(mean, inds[1])
 
 P, d, Q = np.linalg.svd(ratings2 - mean, full_matrices=False)
 d = np.diag(d)
 P = P[:, 0:k]
 Q = Q[0:k, :]
 d = d[0:k, 0:k]
 d = np.sqrt(d)
 Pd = np.dot(P, d)
 dQ = np.dot(d, Q)
 PdQ = np.dot(Pd, dQ) + mean
 
 if rounded:
  return PdQ.round()
 else:
  return PdQ
