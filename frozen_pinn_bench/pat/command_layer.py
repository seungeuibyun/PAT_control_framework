"""Physics-informed command layer: a Frozen-PINN in time that solves (P1) under the FSM dynamics.

tau = 0 is the instant the new command reaches the mirror (after the command latency; the
nominal state there follows exactly from the previous command). Over tau in [0, T_h], where the
new command u is held, each axis is

    theta_a(tau) = theta0_a + dtheta0_a tau + tau^2 sum_j c_aj tanh(alpha_j tau / T_h + beta_j),

so the initial angle and rate hold exactly. The frozen temporal features (alpha_j, beta_j) are
sampled once. The coefficients c and the command u are the least-squares solution of
  * the FSM ODE residual  (theta'' + 2 zeta w theta' + w^2 theta - w^2 u) / w^2  at collocation points,
  * the pointing residual  (J_F theta(tau) + d_hat(tau) - p_tar) / lever  at the same points,
  * a small command-change penalty (u - u_prev).
All terms are linear, so u is a fixed gain applied to (theta0, dtheta0, u_prev, p_tar, d_hat samples),
computed once per controller.
"""
import numpy as np


class PINNCommandLayer:
    def __init__(self, spec, gain_model, cfg):
        dt = spec.dt
        self.horizon = cfg['hold_ticks']*spec.tick_steps*dt
        rng = np.random.default_rng(cfg['seed'])
        J = cfg['features']
        alpha = rng.uniform(-cfg['feature_scale'], cfg['feature_scale'], J)
        beta = -alpha*rng.uniform(0, 1, J)
        tau = np.linspace(0, self.horizon, cfg['collocation'])
        s = tau[:, None]/self.horizon
        phi = np.tanh(alpha*s+beta)
        dphi = alpha*(1-phi**2)/self.horizon
        ddphi = -2*alpha**2*phi*(1-phi**2)/self.horizon**2
        t = tau[:, None]
        g = t**2*phi
        dg = 2*t*phi+t**2*dphi
        ddg = 2*phi+4*t*dphi+t**2*ddphi
        self.tau, self.g = tau, g
        omega, zeta = spec.omega, spec.fsm.damping
        lever = spec.psd_lever
        jac = lever*np.asarray(gain_model, dtype=float)
        nc = len(tau)
        n_unknown = 2*(J+1)
        # Known-input vector b = [theta0(2), dtheta0(2), u_prev(2), p_tar(2), d_hat x (nc), d_hat y (nc)].
        n_known = 8+2*nc
        mu, rho = np.sqrt(cfg['physics_weight']), np.sqrt(cfg['rate_weight'])
        # Pointing is scored on the first point_ticks command periods (the receding horizon issues a
        # new command every T_u); the ODE is enforced over the whole hold horizon.
        point_horizon = cfg.get('point_ticks', cfg['hold_ticks'])*spec.tick_steps*dt
        point = (tau <= point_horizon+1e-15).astype(float)
        rows_a, rows_f = [], []
        for a in range(2):                                   # FSM ODE residual (fsm_dynamics)
            w = omega[a]
            A = np.zeros((nc, n_unknown))
            F = np.zeros((nc, n_known))
            base = a*(J+1)
            A[:, base:base+J] = mu*(ddg+2*zeta*w*dg+w**2*g)/w**2
            A[:, base+J] = -mu
            F[:, a] = -mu                                    # residual = A x - F b
            F[:, 2+a] = -mu*(2*zeta/w+tau)
            rows_a.append(A)
            rows_f.append(F)
        for a in range(2):                                   # pointing residual of (P1)
            A = np.zeros((nc, n_unknown))
            F = np.zeros((nc, n_known))
            for b in range(2):
                A[:, b*(J+1):b*(J+1)+J] = jac[a, b]*g/lever
                F[:, b] = -jac[a, b]/lever
                F[:, 2+b] = -jac[a, b]*tau/lever
            F[:, 6+a] = 1.0/lever
            F[:, 8+a*nc:8+(a+1)*nc] = -np.eye(nc)/lever
            A *= point[:, None]
            F *= point[:, None]
            rows_a.append(A)
            rows_f.append(F)
        for a in range(2):                                   # command-change penalty
            A = np.zeros((1, n_unknown))
            F = np.zeros((1, n_known))
            A[0, a*(J+1)+J] = rho
            F[0, 4+a] = rho
            rows_a.append(A)
            rows_f.append(F)
        A, F = np.vstack(rows_a), np.vstack(rows_f)
        scale = 1/np.maximum(np.linalg.norm(A, axis=0), 1e-300)   # column equilibration
        solution = np.linalg.lstsq(A*scale, F, rcond=1e-12)[0]*scale[:, None]
        self.K = solution                                    # x = K b
        self.K_u = solution[[J, 2*J+1]]                      # rows of u_x, u_y
        self.J = J
        self.n_point = nc
        self.conditioning = float(np.linalg.cond(A*scale))

    def known_vector(self, theta0, dtheta0, u_prev, target, d_hat):
        """theta0, dtheta0, u_prev, target: (B, 2); d_hat: (B, n_point, 2) at self.tau."""
        return np.concatenate([theta0, dtheta0, u_prev, target, d_hat[:, :, 0], d_hat[:, :, 1]], axis=1)

    def command(self, theta0, dtheta0, u_prev, target, d_hat):
        return self.known_vector(theta0, dtheta0, u_prev, target, d_hat)@self.K_u.T

    def trajectory(self, theta0, dtheta0, u_prev, target, d_hat):
        """PINN angle on the collocation grid (B, nc, 2) and the command (B, 2)."""
        x = self.known_vector(theta0, dtheta0, u_prev, target, d_hat)@self.K.T
        J = self.J
        out = []
        for a in range(2):
            c = x[:, a*(J+1):a*(J+1)+J]
            out.append(theta0[:, a, None]+dtheta0[:, a, None]*self.tau[None]+c@self.g.T)
        return np.stack(out, axis=2), x[:, [J, 2*J+1]]
