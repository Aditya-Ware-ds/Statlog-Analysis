"""Gaussian-process classification: Dirichlet-based GP classifier (GPD).

Milios, Camoriano, Michiardi, Rosasco & Filippone, "Dirichlet-based Gaussian
Processes for Large-scale Calibrated Classification", NeurIPS 2018.

Each one-hot label vector is read as one draw from a Dirichlet distribution
with concentration ``alpha_eps + y``. Its log-normal moment-matched
approximation turns classification into C exact GP regressions with known
heteroscedastic noise:

    alpha_ic   = alpha_eps + [y_i = c]
    sigma2_ic  = log(1 / alpha_ic + 1)
    target_ic  = log(alpha_ic) - sigma2_ic / 2

The C latent GPs share one kernel, whose hyper-parameters maximise the summed
log marginal likelihood (type-II maximum likelihood, analytic gradients,
L-BFGS-B). Class probabilities are the Monte-Carlo mean of
softmax(f) with f drawn from the C Gaussian predictive marginals.

Kernel: stationary RBF or Matern-5/2 on standardised features with *grouped*
ARD -- features in the same group share a length-scale. An optional finite
group of feature permutations makes the kernel invariant,
k_inv(x, x') = mean_t k(x, t x'), which is positive semi-definite whenever the
base kernel is invariant under the same permutations (the groups must be
unions of permutation orbits). For StatLog this is the dihedral group of the
3x3 pixel neighbourhood (the class label belongs to the centre pixel, so the
neighbourhood's orientation is irrelevant).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize

from .data import N_BANDS, N_PIXELS, Standardizer

SQRT5 = np.sqrt(5.0)


# ----------------------------------------------------------- feature structure
def dihedral_pixel_permutations() -> list[np.ndarray]:
    """The 8 symmetries of the 3x3 grid as permutations of the 9 pixel positions."""
    grid = np.arange(N_PIXELS).reshape(3, 3)
    perms = []
    for k in range(4):
        g = np.rot90(grid, k)
        perms += [g.ravel(), np.fliplr(g).ravel()]
    return [np.array(p) for p in perms]


def statlog_feature_permutations() -> list[np.ndarray]:
    """Dihedral symmetries acting on the 36 StatLog features (pixel-major, 4 bands per pixel)."""
    return [np.array([N_BANDS * p[q] + b for q in range(N_PIXELS) for b in range(N_BANDS)])
            for p in dihedral_pixel_permutations()]


def statlog_groups(kind: str) -> np.ndarray:
    """Length-scale groups for the 36 StatLog features.

    ``"ard"``: one per feature; ``"band_orbit"``: one per (band, pixel orbit)
    where the orbits of the dihedral group are centre / edges / corners (12
    groups); ``"band"``: one per band; ``"iso"``: a single length-scale.
    """
    pixel = np.repeat(np.arange(N_PIXELS), N_BANDS)
    band = np.tile(np.arange(N_BANDS), N_PIXELS)
    if kind == "ard":
        return np.arange(N_PIXELS * N_BANDS)
    if kind == "band_orbit":
        orbit = np.where(pixel == 4, 0, np.where(pixel % 2 == 1, 1, 2))
        return band * 3 + orbit
    if kind == "band":
        return band
    if kind == "iso":
        return np.zeros(N_PIXELS * N_BANDS, dtype=int)
    raise ValueError(kind)


def contiguous_groups(n_features: int, n_groups: int) -> np.ndarray:
    """Split contiguous spectral bands into ``n_groups`` blocks sharing a length-scale."""
    out = np.empty(n_features, dtype=int)
    for g, idx in enumerate(np.array_split(np.arange(n_features), min(n_groups, n_features))):
        out[idx] = g
    return out


# ------------------------------------------------------------------- kernel
class GroupedKernel:
    """s2 * phi(r), r^2 = sum_g ||x_g - x'_g||^2 / l_g^2, averaged over feature permutations."""

    def __init__(self, groups: np.ndarray, kind: str = "rbf", permutations: list[np.ndarray] | None = None):
        if kind not in ("rbf", "matern52"):
            raise ValueError(kind)
        self.groups = np.asarray(groups, dtype=int)  # -1 = feature not used by this kernel
        self.n_groups = int(self.groups.max()) + 1
        self.kind = kind
        self.perms = [np.arange(len(self.groups))] if permutations is None else [np.asarray(p) for p in permutations]
        for p in self.perms:
            if not np.array_equal(self.groups[p], self.groups):
                raise ValueError("every permutation must map each length-scale group onto itself")
        self.members = [np.flatnonzero(self.groups == g) for g in range(self.n_groups)]

    @property
    def n_params(self) -> int:
        return self.n_groups + 1  # log length-scales + log signal variance

    def _sqdist(self, A: np.ndarray, B: np.ndarray, g: int) -> np.ndarray:
        idx = self.members[g]
        a, b = A[:, idx], B[:, idx]
        d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
        return np.maximum(d, 0.0)

    def _phi(self, r2: np.ndarray) -> np.ndarray:
        if self.kind == "rbf":
            return np.exp(-0.5 * r2)
        r = np.sqrt(r2)
        return (1.0 + SQRT5 * r + 5.0 / 3.0 * r2) * np.exp(-SQRT5 * r)

    def _dphi_factor(self, r2: np.ndarray) -> np.ndarray:
        """F with d k / d log l_g = s2 * F * D_g / l_g^2."""
        if self.kind == "rbf":
            return np.exp(-0.5 * r2)
        r = np.sqrt(r2)
        return 5.0 / 3.0 * (1.0 + SQRT5 * r) * np.exp(-SQRT5 * r)

    def __call__(self, A: np.ndarray, B: np.ndarray, theta: np.ndarray) -> np.ndarray:
        ls2 = np.exp(2.0 * theta[:-1])
        s2 = np.exp(theta[-1])
        K = np.zeros((len(A), len(B)))
        for p in self.perms:
            Bp = B[:, p]
            r2 = sum(self._sqdist(A, Bp, g) / ls2[g] for g in range(self.n_groups))
            K += self._phi(r2)
        return s2 * K / len(self.perms)

    def diag(self, A: np.ndarray, theta: np.ndarray) -> np.ndarray:
        ls2 = np.exp(2.0 * theta[:-1])
        out = np.zeros(len(A))
        for p in self.perms:
            Ap = A[:, p]
            r2 = sum(((A[:, idx] - Ap[:, idx]) ** 2).sum(1) / ls2[g] for g, idx in enumerate(self.members))
            out += self._phi(r2)
        return np.exp(theta[-1]) * out / len(self.perms)

    # distances for repeated marginal-likelihood evaluations on a fixed X
    def precompute(self, X: np.ndarray, cache: bool = True):
        """Per permutation, the stacked group distance matrices (G, n, n) in float32.

        With ``cache=False`` only X is kept and distances are recomputed at every
        evaluation (O(n^2) memory instead of O(T G n^2)).
        """
        if not cache:
            return ("uncached", X)
        return [np.stack([self._sqdist(X, X[:, p], g) for g in range(self.n_groups)]).astype(np.float32)
                for p in self.perms]

    def gram_and_grad_fn(self, D, theta: np.ndarray):
        ls2 = np.exp(2.0 * theta[:-1])
        s2 = np.exp(theta[-1])
        uncached = isinstance(D, tuple) and D[0] == "uncached"
        if uncached:
            X = D[1]
            dist = lambda t, g: self._sqdist(X, X[:, self.perms[t]], g)  # noqa: E731
            n = len(X)
        else:
            dist = lambda t, g: D[t][g]  # noqa: E731
            n = D[0].shape[1]
        T = len(self.perms)
        K = np.zeros((n, n))
        factors = []
        for t in range(T):
            r2 = np.zeros((n, n))
            for g in range(self.n_groups):
                r2 += dist(t, g) / ls2[g]
            K += self._phi(r2)
            factors.append(self._dphi_factor(r2).astype(np.float32))
            del r2
        K *= s2 / T

        def trace_grad(W: np.ndarray) -> np.ndarray:
            """tr(W dK/dtheta) for every hyper-parameter."""
            g = np.zeros(self.n_params)
            for t, F in enumerate(factors):
                WF = W * F
                for k in range(self.n_groups):
                    g[k] += np.sum(WF * dist(t, k)) / ls2[k]
            g[:-1] *= s2 / T
            g[-1] = np.sum(W * K)
            return g

        return K, trace_grad

    def lengthscale_mask(self) -> np.ndarray:
        """True for the log-length-scale entries of theta (False for the log signal variance)."""
        return np.r_[np.ones(self.n_groups, bool), False]

    def initial_theta(self, target_var: float) -> np.ndarray:
        return _initial_theta(self, target_var)


class LinearKernel:
    """Grouped linear (dot-product) kernel k(x, x') = sum_g v_g <x_g, x'_g>, theta = log v_g.

    Used inside a :class:`SumKernel` it adds a Bayesian linear model on the (standardised)
    inputs to a stationary kernel. It has no length-scales, so ``refit``'s length-scale
    multiplier leaves it unchanged.
    """

    def __init__(self, groups: np.ndarray):
        self.groups = np.asarray(groups, dtype=int)  # -1 = feature not used
        self.n_groups = int(self.groups.max()) + 1
        self.members = [np.flatnonzero(self.groups == g) for g in range(self.n_groups)]

    @property
    def n_params(self) -> int:
        return self.n_groups

    def _dot(self, A, B, g):
        idx = self.members[g]
        return A[:, idx] @ B[:, idx].T

    def __call__(self, A, B, theta):
        v = np.exp(theta)
        return sum(v[g] * self._dot(A, B, g) for g in range(self.n_groups))

    def diag(self, A, theta):
        v = np.exp(theta)
        return sum(v[g] * (A[:, idx] ** 2).sum(1) for g, idx in enumerate(self.members))

    def precompute(self, X, cache: bool = True):
        if not cache:
            return ("uncached", X)
        return np.stack([self._dot(X, X, g) for g in range(self.n_groups)]).astype(np.float32)

    def gram_and_grad_fn(self, D, theta):
        v = np.exp(theta)
        if isinstance(D, tuple) and D[0] == "uncached":
            X = D[1]
            dots = [self._dot(X, X, g) for g in range(self.n_groups)]
        else:
            dots = list(D)
        K = sum(v[g] * dots[g] for g in range(self.n_groups))

        def trace_grad(W):
            return np.array([v[g] * np.sum(W * dots[g]) for g in range(self.n_groups)])

        return K, trace_grad

    def lengthscale_mask(self) -> np.ndarray:
        return np.zeros(self.n_groups, bool)

    def initial_theta(self, target_var: float) -> np.ndarray:
        used = sum(len(m) for m in self.members)
        return np.full(self.n_groups, np.log(target_var / max(used, 1)))


class SumKernel:
    """k_1 + ... + k_N with concatenated hyper-parameters (theta = [theta_1, ..., theta_N])."""

    def __init__(self, *kernels):
        if len(kernels) < 2:
            raise ValueError("a sum kernel needs at least two terms")
        self.kernels = list(kernels)
        self.n_groups = sum(k.n_groups for k in self.kernels)

    @property
    def k1(self) -> GroupedKernel:
        return self.kernels[0]

    @property
    def k2(self) -> GroupedKernel:
        return self.kernels[1]

    @property
    def n_params(self) -> int:
        return sum(k.n_params for k in self.kernels)

    def _split(self, theta):
        out, i = [], 0
        for k in self.kernels:
            out.append(theta[i:i + k.n_params])
            i += k.n_params
        return out

    def __call__(self, A, B, theta):
        return sum(k(A, B, t) for k, t in zip(self.kernels, self._split(theta)))

    def diag(self, A, theta):
        return sum(k.diag(A, t) for k, t in zip(self.kernels, self._split(theta)))

    def precompute(self, X, cache: bool = True):
        return tuple(k.precompute(X, cache) for k in self.kernels)

    def gram_and_grad_fn(self, D, theta):
        parts = [k.gram_and_grad_fn(d, t) for k, d, t in zip(self.kernels, D, self._split(theta))]
        K = sum(p[0] for p in parts)
        return K, lambda W: np.concatenate([p[1](W) for p in parts])

    def initial_theta(self, target_var: float) -> np.ndarray:
        share = target_var / len(self.kernels)
        return np.concatenate([k.initial_theta(share) for k in self.kernels])

    def lengthscale_mask(self) -> np.ndarray:
        """True for the log-length-scale entries of theta (False for variances)."""
        return np.concatenate([k.lengthscale_mask() for k in self.kernels])


def _initial_theta(k: GroupedKernel, target_var: float) -> np.ndarray:
    sizes = np.array([len(m) for m in k.members])
    return np.r_[0.5 * np.log(np.maximum(sizes, 1)), np.log(target_var)]


# ---------------------------------------------------------------- classifier
class GPDirichletClassifier:
    """Dirichlet-based Gaussian-process classifier with a grouped-ARD (optionally invariant) kernel.

    Parameters
    ----------
    representation : input standardisation, "standard" (alias "full36") or "log36".
    groups : length-scale group of each feature (after standardisation);
        ``None`` gives one length-scale per feature.
    kernel : "rbf", "matern52", or a ``GroupedKernel`` / ``SumKernel`` instance.
    permutations : optional feature permutations the kernel is made invariant to.
    alpha_eps : Dirichlet concentration of the non-observed classes.
    cache : cache the pairwise group distances during optimisation (fast,
        O(T G n^2) memory); set False for large n with invariant kernels.
    theta : fixed hyper-parameters (no optimisation); theta_init : optimiser start.
    features : optional feature map with ``fit``/``transform`` used instead of
        ``representation`` (e.g. ``CompositeFeatures``).
    rule : class-probability rule -- "mc" (Monte-Carlo mean of softmax(f)) or
        "lognormal" (deterministic, p_c proportional to exp(mu_c + var_c / 2),
        the mean of the Gamma-like variable each latent approximates).
    temperature : the latent functions are divided by it before the rule is applied
        (calibration only: it does not change the "lognormal" decisions).
    n_opt : at most this many (class-stratified) training samples are used to
        optimise the hyper-parameters; the final model uses all of them.
    """

    def __init__(self, representation: str = "standard", groups=None, kernel: str = "rbf", permutations=None,
                 alpha_eps: float = 0.01, n_opt: int = 2000, max_iter: int = 100, n_mc: int = 256,
                 jitter: float = 1e-6, random_state: int = 0, theta=None,
                 cache: bool = True, theta_init=None, features=None, rule: str = "mc", temperature: float = 1.0):
        self.representation = representation
        self.groups = groups
        self.kernel = kernel
        self.permutations = permutations
        self.alpha_eps = alpha_eps
        self.n_opt = n_opt
        self.max_iter = max_iter
        self.n_mc = n_mc
        self.jitter = jitter
        self.random_state = random_state
        self.theta = theta
        self.cache = cache
        self.theta_init = theta_init
        self.features = features
        if rule not in ("mc", "lognormal"):
            raise ValueError("rule must be 'mc' or 'lognormal'")
        self.rule = rule
        self.temperature = temperature
        self.prior_ratio_ = None

    # -- label transformation
    def _noise_levels(self, alpha_eps: float) -> tuple[float, float]:
        """Log-normal noise variances of the observed (on) and non-observed (off) classes."""
        return float(np.log(1.0 / (1.0 + alpha_eps) + 1.0)), float(np.log(1.0 / alpha_eps + 1.0))

    def _targets(self, y: np.ndarray, alpha_eps: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        a_eps = self.alpha_eps if alpha_eps is None else alpha_eps
        Y = (y[:, None] == self.classes_[None, :]).astype(float)
        alpha = a_eps + Y
        s2 = np.log(1.0 / alpha + 1.0)
        return np.log(alpha) - 0.5 * s2, s2

    # -- shared-noise (Woodbury) algebra
    #
    # The noise of class c is s_off everywhere except on the samples of class
    # c, where it is s_on < s_off. Hence A_c = B + P_c (s_on - s_off) P_c^T with
    # B = K + s_off I shared by all classes, and by the Woodbury identity
    #   A_c^{-1} = B^{-1} + G_c N_c^{-1} G_c^T,   G_c = B^{-1} P_c,
    #   N_c = I / (s_off - s_on) - P_c^T B^{-1} P_c   (positive definite),
    #   log|A_c| = log|B| + n_c log(s_off - s_on) + log|N_c|.
    # One n x n factorisation serves all classes.
    def _woodbury(self, K: np.ndarray, members: list[np.ndarray]):
        n = len(K)
        B = K.copy()
        B[np.diag_indices(n)] += self.s_off_ + self.jitter
        LB = cho_factor(B, lower=True, check_finite=False)
        Binv = cho_solve(LB, np.eye(n), check_finite=False)
        logdet_B = 2.0 * np.log(np.diag(LB[0])).sum()
        gap = self.s_off_ - self.s_on_
        LN = [cho_factor(np.eye(len(idx)) / gap - Binv[np.ix_(idx, idx)], lower=True, check_finite=False)
              for idx in members]
        return LB, Binv, logdet_B, LN

    def _neg_lml(self, theta, D, T, members):
        n, C = T.shape
        K, trace_grad = self.kernel_.gram_and_grad_fn(D, theta)
        try:
            _, Binv, logdet_B, LN = self._woodbury(K, members)
        except np.linalg.LinAlgError:
            return 1e25, np.zeros_like(theta)
        gap = self.s_off_ - self.s_on_
        W = -C * Binv
        lml = 0.0
        for c, idx in enumerate(members):
            r = T[:, c] - self.means_[c]
            b = Binv @ r
            G = Binv[:, idx]
            a = b + G @ cho_solve(LN[c], b[idx], check_finite=False)
            logdet_A = logdet_B + len(idx) * np.log(gap) + 2.0 * np.log(np.diag(LN[c][0])).sum()
            lml += -0.5 * r @ a - 0.5 * logdet_A - 0.5 * n * np.log(2 * np.pi)
            W += np.outer(a, a) - G @ cho_solve(LN[c], G.T, check_finite=False)
        return -lml, -0.5 * trace_grad(W)

    def log_marginal_likelihood(self, theta=None) -> float:
        """Summed log marginal likelihood (over classes) on the hyper-parameter optimisation subset."""
        if getattr(self, "_D_opt", None) is None:
            raise RuntimeError("the training cache was released; the evidence at the learned hyper-parameters "
                               "is kept in log_evidence_ when the model was fitted by fit_gp_v2")
        theta = self.theta_ if theta is None else np.asarray(theta)
        return -self._neg_lml(theta, self._D_opt, self._T_opt, self._members_opt)[0]

    def release_training_cache(self) -> "GPDirichletClassifier":
        """Drop the pairwise distances kept for hyper-parameter optimisation (O(groups x n^2) memory).

        Prediction, ``refit`` and the selection steps do not need them; ``log_marginal_likelihood`` does.
        """
        self._D_opt = None
        return self

    def _subset(self, y: np.ndarray, rng) -> np.ndarray:
        if len(y) <= self.n_opt:
            return np.arange(len(y))
        idx = []
        for c in self.classes_:
            pool = np.flatnonzero(y == c)
            k = max(2, int(round(self.n_opt * len(pool) / len(y))))
            idx.append(rng.choice(pool, size=min(k, len(pool)), replace=False))
        return np.sort(np.concatenate(idx))

    def _members(self, y: np.ndarray) -> list[np.ndarray]:
        return [np.flatnonzero(y == c) for c in self.classes_]

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None):
        y = np.asarray(y)
        self.rep_ = (self.features if self.features is not None else Standardizer(self.representation)).fit(X)
        Z = self.rep_.transform(X)
        self.classes_ = np.unique(y)
        self.s_on_, self.s_off_ = self._noise_levels(self.alpha_eps)
        if isinstance(self.kernel, (GroupedKernel, SumKernel, LinearKernel)):
            self.kernel_ = self.kernel
        else:
            groups = np.arange(Z.shape[1]) if self.groups is None else np.asarray(self.groups)
            self.kernel_ = GroupedKernel(groups, self.kernel, self.permutations)
        T, _ = self._targets(y)
        self.means_ = T.mean(axis=0)

        rng = np.random.default_rng(self.random_state)
        sub = self._subset(y, rng)
        self._D_opt = self.kernel_.precompute(Z[sub], self.cache)
        self._T_opt = T[sub]
        self._members_opt = self._members(y[sub])
        if self.theta is not None:
            self.theta_ = np.asarray(self.theta, dtype=float)
            self.opt_result_ = None
        else:
            target_var = T.var(axis=0).mean()
            if self.theta_init is not None:
                theta0 = np.asarray(self.theta_init, dtype=float)
            else:
                theta0 = self.kernel_.initial_theta(target_var)
            bounds = [(np.log(1e-2), np.log(1e3))] * self.kernel_.n_params
            res = minimize(self._neg_lml, theta0, args=(self._D_opt, self._T_opt, self._members_opt), jac=True,
                           method="L-BFGS-B", bounds=bounds, options={"maxiter": self.max_iter})
            self.theta_ = res.x
            self.opt_result_ = res
        self.theta_ml_ = self.theta_.copy()
        self.y_ = y
        self._fit_final(Z, y, T)
        return self

    def _lengthscale_mask(self) -> np.ndarray:
        return self.kernel_.lengthscale_mask()

    def refit(self, alpha_eps: float | None = None, lengthscale_scale: float = 1.0) -> "GPDirichletClassifier":
        """Re-condition on the training data with a new alpha_eps and/or all length-scales multiplied
        by ``lengthscale_scale`` (relative to the maximum-likelihood values); no re-optimisation."""
        if alpha_eps is not None:
            self.alpha_eps = alpha_eps
        self.s_on_, self.s_off_ = self._noise_levels(self.alpha_eps)
        self.theta_ = self.theta_ml_ + np.log(lengthscale_scale) * self._lengthscale_mask()
        T, _ = self._targets(self.y_)
        self.means_ = T.mean(axis=0)
        self._fit_final(self.Z_, self.y_, T)
        return self

    def _fit_final(self, Z: np.ndarray, y: np.ndarray, T: np.ndarray) -> None:
        self.Z_ = Z
        self.members_ = self._members(y)
        K = self.kernel_(Z, Z, self.theta_)
        self.LB_, Binv, _, self.LN_ = self._woodbury(K, self.members_)
        alpha = []
        for c, idx in enumerate(self.members_):
            b = Binv @ (T[:, c] - self.means_[c])
            alpha.append(b + Binv[:, idx] @ cho_solve(self.LN_[c], b[idx], check_finite=False))
        self.alpha_ = np.stack(alpha, axis=1)

    def refit_alpha_eps(self, X: np.ndarray, y: np.ndarray, alpha_eps: float) -> "GPDirichletClassifier":
        """Re-fit with a new alpha_eps keeping the learned kernel hyper-parameters."""
        y = np.asarray(y)
        self.alpha_eps = alpha_eps
        self.s_on_, self.s_off_ = self._noise_levels(alpha_eps)
        T, _ = self._targets(y)
        self.means_ = T.mean(axis=0)
        self._fit_final(self.rep_.transform(X), y, T)
        return self

    def predict_latent(self, X: np.ndarray, chunk: int = 4000) -> tuple[np.ndarray, np.ndarray]:
        """Predictive means and variances of the C latent functions."""
        Z = self.rep_.transform(X)
        C = len(self.classes_)
        mu = np.empty((len(Z), C))
        var = np.empty((len(Z), C))
        for s in range(0, len(Z), chunk):
            Zc = Z[s:s + chunk]
            Ks = self.kernel_(Zc, self.Z_, self.theta_)
            kss = self.kernel_.diag(Zc, self.theta_)
            mu[s:s + chunk] = self.means_ + Ks @ self.alpha_
            U = cho_solve(self.LB_, Ks.T, check_finite=False)       # B^{-1} k_*
            base = kss - (Ks.T * U).sum(0)
            for c, idx in enumerate(self.members_):
                w = solve_triangular(self.LN_[c][0], U[idx], lower=True, check_finite=False)
                var[s:s + chunk, c] = np.maximum(base - (w * w).sum(0), 1e-12)
        return mu, var

    def probabilities_from_latent(self, mu: np.ndarray, var: np.ndarray, chunk: int = 2000) -> np.ndarray:
        """Class probabilities from latent means/variances under ``self.rule`` (and prior ratio, if set)."""
        T = self.temperature
        if self.rule == "lognormal":
            logit = (mu + 0.5 * var) / T
            logit -= logit.max(axis=1, keepdims=True)
            P = np.exp(logit)
            P /= P.sum(axis=1, keepdims=True)
        else:
            rng = np.random.default_rng(self.random_state)
            P = np.empty_like(mu)
            sd = np.sqrt(var)
            for s in range(0, len(mu), chunk):
                m, d = mu[s:s + chunk], sd[s:s + chunk]
                f = (m[:, :, None] + d[:, :, None] * rng.standard_normal((*m.shape, self.n_mc))) / T
                f -= f.max(axis=1, keepdims=True)
                e = np.exp(f)
                P[s:s + chunk] = (e / e.sum(axis=1, keepdims=True)).mean(axis=2)
        if self.prior_ratio_ is not None:
            P = P * self.prior_ratio_[None, :]
            P /= P.sum(axis=1, keepdims=True)
        return P

    def predict_proba(self, X: np.ndarray, chunk: int = 2000) -> np.ndarray:
        mu, var = self.predict_latent(X)
        return self.probabilities_from_latent(mu, var, chunk)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def scores(self, X) -> dict[str, np.ndarray]:
        P = self.predict_proba(X)
        return {"pred": self.classes_[P.argmax(axis=1)], "max_prob": P.max(axis=1)}

    @property
    def lengthscales_(self) -> np.ndarray:
        """Length-scales (for a sum kernel: those of all its stationary terms, in order)."""
        return np.exp(self.theta_[self._lengthscale_mask()])

    @property
    def signal_variance_(self) -> float:
        """Signal variance of the stationary term(s) (summed for a sum kernel)."""
        if isinstance(self.kernel_, SumKernel):
            return float(sum(np.exp(t[-1]) for k, t in zip(self.kernel_.kernels, self.kernel_._split(self.theta_))
                             if isinstance(k, GroupedKernel)))
        return float(np.exp(self.theta_[-1]))
