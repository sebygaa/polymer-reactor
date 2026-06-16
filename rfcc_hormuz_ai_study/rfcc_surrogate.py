#!/usr/bin/env python3
"""
rfcc_surrogate.py
Pure-NumPy MLP surrogate of the RFCC heat-balanced unit model.
Trained in rfcc_step4_ai_surrogate_v01.py, consumed as a ~1000x faster
stand-in for rfcc_model.solve_unit by the AI optimisers in Step 5.

Inputs  (5): [ROT_C, T_pre_C, Qcool_frac, CCR, feed_frac]
Outputs (8): [y_hs, y_lco, y_gaso, y_gas, y_coke, T_rgn_C, CO, T_out_C]
"""

import numpy as np

X_NAMES = ['ROT_C', 'T_pre_C', 'Qcool_frac', 'CCR', 'feed_frac']
Y_NAMES = ['y_hs', 'y_lco', 'y_gaso', 'y_gas', 'y_coke',
           'T_rgn_C', 'CO', 'T_out_C']


class MLP:
    """2-hidden-layer tanh MLP with z-score scaling and Adam training."""

    def __init__(self, n_in, n_out, n_hid=96, seed=0):
        rng = np.random.default_rng(seed)
        s1 = np.sqrt(2.0 / n_in)
        s2 = np.sqrt(2.0 / n_hid)
        self.W1 = rng.normal(0, s1, (n_in, n_hid)); self.b1 = np.zeros(n_hid)
        self.W2 = rng.normal(0, s2, (n_hid, n_hid)); self.b2 = np.zeros(n_hid)
        self.W3 = rng.normal(0, s2, (n_hid, n_out)); self.b3 = np.zeros(n_out)
        self.x_mu = np.zeros(n_in); self.x_sd = np.ones(n_in)
        self.y_mu = np.zeros(n_out); self.y_sd = np.ones(n_out)

    # ----- forward ---------------------------------------------------
    def _fwd(self, Xs):
        H1 = np.tanh(Xs @ self.W1 + self.b1)
        H2 = np.tanh(H1 @ self.W2 + self.b2)
        return H1, H2, H2 @ self.W3 + self.b3

    def predict(self, X):
        X = np.atleast_2d(X)
        Xs = (X - self.x_mu) / self.x_sd
        _, _, Ys = self._fwd(Xs)
        return Ys * self.y_sd + self.y_mu

    # ----- training --------------------------------------------------
    def fit(self, X, Y, X_val=None, Y_val=None, epochs=1500, batch=256,
            lr=1e-3, seed=1, verbose_every=200):
        rng = np.random.default_rng(seed)
        self.x_mu, self.x_sd = X.mean(0), X.std(0) + 1e-12
        self.y_mu, self.y_sd = Y.mean(0), Y.std(0) + 1e-12
        Xs = (X - self.x_mu) / self.x_sd
        Ys = (Y - self.y_mu) / self.y_sd
        params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1a, b2a, eps, t = 0.9, 0.999, 1e-8, 0
        n = len(Xs)
        hist = {'epoch': [], 'train': [], 'val': []}
        for ep in range(epochs):
            idx = rng.permutation(n)
            for s in range(0, n, batch):
                bi = idx[s:s + batch]
                xb, yb = Xs[bi], Ys[bi]
                H1, H2, out = self._fwd(xb)
                err = (out - yb) / len(bi)
                gW3 = H2.T @ err; gb3 = err.sum(0)
                dH2 = (err @ self.W3.T) * (1 - H2 * H2)
                gW2 = H1.T @ dH2; gb2 = dH2.sum(0)
                dH1 = (dH2 @ self.W2.T) * (1 - H1 * H1)
                gW1 = xb.T @ dH1; gb1 = dH1.sum(0)
                grads = [gW1, gb1, gW2, gb2, gW3, gb3]
                t += 1
                for p, g, mi, vi in zip(params, grads, m, v):
                    mi *= b1a; mi += (1 - b1a) * g
                    vi *= b2a; vi += (1 - b2a) * g * g
                    p -= lr * (mi / (1 - b1a ** t)) / \
                        (np.sqrt(vi / (1 - b2a ** t)) + eps)
            if ep % 10 == 0 or ep == epochs - 1:
                tr = float(np.mean((self._fwd(Xs)[2] - Ys) ** 2))
                hist['epoch'].append(ep); hist['train'].append(tr)
                if X_val is not None:
                    Xv = (X_val - self.x_mu) / self.x_sd
                    Yv = (Y_val - self.y_mu) / self.y_sd
                    hist['val'].append(
                        float(np.mean((self._fwd(Xv)[2] - Yv) ** 2)))
                if verbose_every and ep % verbose_every == 0:
                    msg = f'  epoch {ep:5d}  train MSE {tr:.5f}'
                    if X_val is not None:
                        msg += f'  val MSE {hist["val"][-1]:.5f}'
                    print(msg)
        return hist

    # ----- persistence -----------------------------------------------
    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3, x_mu=self.x_mu, x_sd=self.x_sd,
                 y_mu=self.y_mu, y_sd=self.y_sd)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        obj = cls(d['W1'].shape[0], d['W3'].shape[1], d['W1'].shape[1])
        for k in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3',
                  'x_mu', 'x_sd', 'y_mu', 'y_sd'):
            setattr(obj, k, d[k])
        return obj
