"""Frozen-PINN disturbance predictor (and its linear special case, M = 0).

Input at PSD frame k: the last N pseudo-open-loop samples d_k..d_{k-N+1}
(optical angle, FSM plane), as differences to d_k plus d_k itself.
Output: d_{k+h} - d_k for h = 1..H frames (both axes). Labels are future
PSD frames only, so the same fit can be repeated on bench data.

Frozen model = Frozen-PINN training principle applied to the time axis:
hidden tanh weights are sampled once from data pairs (SWIM-style, no
gradient descent) and frozen; only the linear readout is solved in closed
form. It contains the linear model as a subset (the raw inputs are kept).
Online refit re-solves the readout on a recent window, pulled towards the
offline solution (prior-centred ridge), in one linear solve.
"""
import numpy as np

SWIM_SCALE = np.arctanh(0.5)


class ContextModel:
    def __init__(self, history, horizon, features=0, seed=0):
        self.history, self.horizon, self.features_count = history, horizon, features
        self.seed = seed
        self.weights = self.bias = self.beta = None
        self.sigma_diff = self.sigma_abs = None
        self.fit_report = {}

    @property
    def size(self):
        return 2*self.history+self.features_count+1

    def inputs(self, window):
        """window (..., N, 2) oldest -> newest."""
        latest = window[..., -1, :]
        diffs = (window[..., :-1, :]-latest[..., None, :])/self.sigma_diff
        return np.concatenate([diffs.reshape(*diffs.shape[:-2], -1), latest/self.sigma_abs], axis=-1)

    def targets(self, latest, future):
        """future (..., H, 2) -> scaled increments (..., 2H)."""
        return ((future-latest[..., None, :])/self.sigma_diff).reshape(*future.shape[:-2], -1)

    def increments(self, outputs):
        return outputs.reshape(*outputs.shape[:-1], self.horizon, 2)*self.sigma_diff

    def features(self, u):
        parts = [u]
        if self.features_count:
            parts.append(np.tanh(u@self.weights+self.bias))
        parts.append(np.ones(u.shape[:-1]+(1,)))
        return np.concatenate(parts, axis=-1)

    def samples(self, sequences):
        """All (input, target) pairs from a list of (F, 2) sequences."""
        n, h = self.history, self.horizon
        us, ys = [], []
        for seq in sequences:
            count = len(seq)-n-h+1
            if count <= 0:
                continue
            idx = np.arange(count)[:, None]+np.arange(n)[None, :]
            window = seq[idx]
            future = seq[np.arange(count)[:, None]+n-1+np.arange(1, h+1)[None, :]]
            us.append(self.inputs(window))
            ys.append(self.targets(window[:, -1], future))
        return np.concatenate(us), np.concatenate(ys)

    def _sample_hidden(self, u, y, candidates, rng):
        i = rng.integers(0, len(u), candidates)
        j = rng.integers(0, len(u), candidates)
        du = u[j]-u[i]
        dist = np.linalg.norm(du, axis=1)
        keep = dist > 1e-9
        i, j, du, dist = i[keep], j[keep], du[keep], dist[keep]
        ratio = np.linalg.norm(y[j]-y[i], axis=1)/dist
        prob = ratio/ratio.sum()
        pick = rng.choice(len(i), self.features_count, replace=False, p=prob)
        w = 2*SWIM_SCALE*du[pick]/dist[pick, None]**2
        b = -np.sum(w*(u[i[pick]]+u[j[pick]])/2, axis=1)
        self.weights, self.bias = w.T.copy(), b

    def fit(self, train, validation, ridge_grid, candidates=40000):
        rng = np.random.default_rng(self.seed)
        diffs = np.concatenate([np.diff(s, axis=0) for s in train])
        self.sigma_diff = float(np.std(diffs))
        self.sigma_abs = float(np.std(np.concatenate(train)))
        u, y = self.samples(train)
        if self.features_count:
            self._sample_hidden(u, y, candidates, rng)
        gram = np.zeros((self.size, self.size))
        cross = np.zeros((self.size, y.shape[1]))
        for start in range(0, len(u), 8192):
            x = self.features(u[start:start+8192])
            gram += x.T@x
            cross += x.T@y[start:start+8192]
        uv, yv = self.samples(validation)
        xv = self.features(uv)
        penalty = np.eye(self.size)
        penalty[-1, -1] = 0.0
        scale = np.trace(gram)/self.size
        best = None
        scores = {}
        for lam in ridge_grid:
            beta = np.linalg.solve(gram+lam*scale*penalty, cross)
            err = ((xv@beta-yv).reshape(len(yv), self.horizon, 2))*self.sigma_diff
            mse = float(np.mean(err[:, :2]**2))
            scores[lam] = mse
            if best is None or mse < best[0]:
                best = (mse, lam, beta)
        self.beta = best[2]
        err = ((xv@self.beta-yv).reshape(len(yv), self.horizon, 2))*self.sigma_diff
        self.fit_report = dict(ridge=best[1], validation_mse_h1_h2=best[0], ridge_scores=scores,
                               validation_rmse_by_horizon_rad=np.sqrt(np.mean(err**2, axis=(0, 2))).tolist(),
                               train_samples=len(u), parameters=self.size)
        return self


def save_model(model, path):
    """Store a fitted predictor (frozen features, readout, scalings) for deployment or replay."""
    arrays = dict(history=model.history, horizon=model.horizon, features=model.features_count, seed=model.seed,
                  beta=model.beta, sigma_diff=model.sigma_diff, sigma_abs=model.sigma_abs)
    if model.features_count:
        arrays.update(weights=model.weights, bias=model.bias)
    np.savez(path, **arrays)


def load_model(path):
    data = np.load(path)
    model = ContextModel(int(data['history']), int(data['horizon']), int(data['features']), int(data['seed']))
    model.beta, model.sigma_diff, model.sigma_abs = data['beta'], float(data['sigma_diff']), float(data['sigma_abs'])
    if model.features_count:
        model.weights, model.bias = data['weights'], data['bias']
    return model


class OnlineReadout:
    """Per-episode readouts; window refit pulled to the offline beta (prior-centred ridge)."""
    def __init__(self, model, batch, window, every, prior):
        self.model, self.window, self.every, self.prior = model, window, every, prior
        self.beta = np.repeat(model.beta[None], batch, axis=0)
        self.x = np.zeros((batch, window, model.size))
        self.y = np.zeros((batch, window, model.beta.shape[1]))
        self.filled = 0
        self.added = 0
        self.refits = 0

    def predict(self, x):
        return np.einsum('bp,bpo->bo', x, self.beta)

    def add(self, x, y):
        slot = self.added % self.window
        self.x[:, slot], self.y[:, slot] = x, y
        self.added += 1
        self.filled = min(self.filled+1, self.window)
        if self.prior is not None and self.added % self.every == 0 and self.filled >= self.every:
            xs, ys = self.x[:, :self.filled], self.y[:, :self.filled]
            gram = np.matmul(xs.transpose(0, 2, 1), xs)
            rhs = np.matmul(xs.transpose(0, 2, 1), ys)+self.prior*self.model.beta[None]
            gram += self.prior*np.eye(self.model.size)[None]
            self.beta = np.linalg.solve(gram, rhs)
            self.refits += 1


def replay_online(model, sequences, window, every, prior):
    """Open-loop replay of online prediction on equal-length sequences; returns h=1,2 MSE."""
    seqs = np.stack(sequences)
    batch, frames, _ = seqs.shape
    n, h = model.history, model.horizon
    online = OnlineReadout(model, batch, window, every, prior)
    errors = []
    pending = {}
    for k in range(n-1, frames):
        x = model.features(model.inputs(seqs[:, k-n+1:k+1]))
        pending[k] = x
        if k+h < frames:
            pred = model.increments(online.predict(x))
            truth = seqs[:, k+1:k+h+1]-seqs[:, k, None]
            errors.append((pred-truth)[:, :2])
        done = k-h
        if done in pending:
            target = model.targets(seqs[:, done], seqs[:, done+1:done+h+1])
            online.add(pending.pop(done), target)
    return float(np.mean(np.concatenate(errors, axis=1)**2))
