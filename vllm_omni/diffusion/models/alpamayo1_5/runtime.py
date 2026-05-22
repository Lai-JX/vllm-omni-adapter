from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Literal, Protocol

import einops
import torch
from torch import nn

logger = logging.getLogger(__name__)


class StepFn(Protocol):
    def __call__(self, *, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor: ...


class ActionSpace(ABC, nn.Module):
    @abstractmethod
    def traj_to_action(
        self,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
        traj_future_xyz: torch.Tensor,
        traj_future_rot: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor: ...

    @abstractmethod
    def action_to_traj(
        self,
        action: torch.Tensor,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def get_action_space_dims(self) -> tuple[int, ...]: ...


def so3_to_yaw_torch(rot_mat: torch.Tensor) -> torch.Tensor:
    return torch.atan2(rot_mat[..., 1, 0], rot_mat[..., 0, 0])


def round_2pi_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def unwrap_angle(phi: torch.Tensor) -> torch.Tensor:
    delta = torch.diff(phi, dim=-1)
    delta = round_2pi_torch(delta)
    return torch.cat([phi[..., :1], phi[..., :1] + torch.cumsum(delta, dim=-1)], dim=-1)


def rotation_matrix_torch(angle: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            torch.stack([torch.cos(angle), -torch.sin(angle)], dim=-1),
            torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1),
        ],
        dim=-2,
    )


def rot_2d_to_3d(rot: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            torch.cat([rot, torch.zeros_like(rot[..., :1])], dim=-1),
            torch.tensor([0.0, 0.0, 1.0], device=rot.device).repeat(rot.shape[:-2] + (1, 1)),
        ],
        dim=-2,
    )


def first_order_D(N: int, lead_shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    D = torch.zeros(*lead_shape, N - 1, N, dtype=dtype, device=device)
    rows = torch.arange(N - 1, device=device)
    D[..., rows, rows] = -1.0
    D[..., rows, rows + 1] = 1.0
    return D


def second_order_D(N: int, lead_shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    D = torch.zeros(*lead_shape, max(N - 2, 0), N, dtype=dtype, device=device)
    rows = torch.arange(max(N - 2, 0), device=device)
    D[..., rows, rows] = -1.0
    D[..., rows, rows + 1] = 2.0
    D[..., rows, rows + 2] = -1.0
    return D


def third_order_D(N: int, lead_shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    D = torch.zeros(*lead_shape, max(N - 3, 0), N, dtype=dtype, device=device)
    rows = torch.arange(max(N - 3, 0), device=device)
    D[..., rows, rows] = -1.0
    D[..., rows, rows + 1] = 3.0
    D[..., rows, rows + 2] = -3.0
    D[..., rows, rows + 3] = 1.0
    return D


@torch.amp.autocast(device_type="cuda", enabled=False)
@torch.no_grad()
@torch._dynamo.disable()
def construct_DTD(
    N: int,
    lead: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    w_smooth1: float | torch.Tensor | None = None,
    w_smooth2: float | torch.Tensor | None = None,
    w_smooth3: float | torch.Tensor | None = None,
    lam: float = 1e-3,
    dt: float = 1.0,
) -> torch.Tensor:
    DTD = torch.zeros(*lead, N, N, dtype=dtype, device=device)
    if w_smooth1 is not None:
        tensor = (
            torch.full((*lead, max(N - 1, 0)), w_smooth1, dtype=dtype, device=device)
            if isinstance(w_smooth1, float)
            else w_smooth1
        )
        D1 = first_order_D(N, lead, device=device, dtype=dtype)
        DTD += lam / dt**2 * einops.einsum(D1 * tensor.unsqueeze(-1), D1, "... i j, ... i k -> ... j k")
    if w_smooth2 is not None:
        tensor = (
            torch.full((*lead, max(N - 2, 0)), w_smooth2, dtype=dtype, device=device)
            if isinstance(w_smooth2, float)
            else w_smooth2
        )
        D2 = second_order_D(N, lead, device=device, dtype=dtype)
        DTD += lam / dt**4 * einops.einsum(D2 * tensor.unsqueeze(-1), D2, "... i j, ... i k -> ... j k")
    if w_smooth3 is not None:
        tensor = (
            torch.full((*lead, max(N - 3, 0)), w_smooth3, dtype=dtype, device=device)
            if isinstance(w_smooth3, float)
            else w_smooth3
        )
        D3 = third_order_D(N, lead, device=device, dtype=dtype)
        DTD += lam / dt**6 * einops.einsum(D3 * tensor.unsqueeze(-1), D3, "... i j, ... i k -> ... j k")
    return DTD


@torch.amp.autocast(device_type="cuda", enabled=False)
@torch.no_grad()
@torch._dynamo.disable()
def solve_single_constraint(
    x_init: torch.Tensor,
    x_target: torch.Tensor,
    w_data: torch.Tensor | None = None,
    w_smooth1: float | torch.Tensor | None = None,
    w_smooth2: float | torch.Tensor | None = None,
    w_smooth3: float | torch.Tensor | None = None,
    lam: float = 1e-3,
    ridge: float = 0.0,
    dt: float = 1.0,
) -> torch.Tensor:
    device, dtype = x_target.device, x_target.dtype
    *lead, N = x_target.shape
    if w_data is None:
        w_data = torch.ones_like(x_target)
    x_init = torch.as_tensor(x_init, dtype=dtype, device=device)
    A_data = torch.eye(N, dtype=dtype, device=device).expand(*lead, N, N)
    Aw_data = A_data * w_data.unsqueeze(-1)
    ATA = einops.einsum(Aw_data, A_data, "... i j, ... i k -> ... j k")
    rhs = einops.einsum(Aw_data, x_target, "... i j, ... i -> ... j")
    DTD = construct_DTD(
        N + 1,
        lead,
        device=device,
        dtype=dtype,
        w_smooth1=w_smooth1,
        w_smooth2=w_smooth2,
        w_smooth3=w_smooth3,
        lam=lam,
        dt=dt,
    )
    rhs -= DTD[..., 1:, 0] * x_init.unsqueeze(-1)
    ridge_term = ridge * torch.eye(N, dtype=dtype, device=device).expand(*lead, N, N)
    lhs = ATA + DTD[..., 1:, 1:] + ridge_term
    L = torch.linalg.cholesky(lhs)
    x = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
    return torch.cat([x_init.unsqueeze(-1), x], dim=-1)


@torch.amp.autocast(device_type="cuda", enabled=False)
@torch.no_grad()
@torch._dynamo.disable()
def solve_xs_eq_y(
    s: torch.Tensor,
    y: torch.Tensor,
    w_data: torch.Tensor | None = None,
    w_smooth1: float | torch.Tensor | None = None,
    w_smooth2: float | torch.Tensor | None = None,
    w_smooth3: float | torch.Tensor | None = None,
    lam: float = 1e-3,
    ridge: float = 0.0,
    dt: float = 1.0,
) -> torch.Tensor:
    device, dtype = y.device, y.dtype
    *lead, N = y.shape
    if w_data is None:
        w_data = torch.ones_like(y)
    A_data = torch.diag_embed(s)
    Aw_data = A_data * w_data.unsqueeze(-1)
    ATA = einops.einsum(Aw_data, A_data, "... i j, ... i k -> ... j k")
    rhs = einops.einsum(Aw_data, y, "... i j, ... i -> ... j")
    DTD = construct_DTD(
        N,
        lead,
        device=device,
        dtype=dtype,
        w_smooth1=w_smooth1,
        w_smooth2=w_smooth2,
        w_smooth3=w_smooth3,
        lam=lam,
        dt=dt,
    )
    L = None
    while L is None:
        try:
            ridge_term = ridge * torch.eye(N, dtype=dtype, device=device).expand(*lead, N, N)
            lhs = ATA + DTD + ridge_term
            rhs_cast = rhs.to(lhs.dtype) if rhs.dtype != lhs.dtype else rhs
            L = torch.linalg.cholesky(lhs)
        except RuntimeError:
            ridge = max(ridge, 1e-8) * 10
    return torch.cholesky_solve(rhs_cast.unsqueeze(-1), L).squeeze(-1)


@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
@torch._dynamo.disable()
def dxy_theta_to_v_without_v0(
    dxy: torch.Tensor,
    theta: torch.Tensor,
    dt: float = 1.0,
    v_lambda: float = 1e-4,
    v_ridge: float = 1e-4,
) -> torch.Tensor:
    *lead, N, _ = dxy.shape
    device, dtype = dxy.device, dxy.dtype
    g = 2 / dt * dxy
    w = torch.ones_like(dxy[..., 0])
    A_data = torch.zeros(*lead, 2 * N, N + 1, dtype=dtype, device=device)
    b_data = g.flatten(start_dim=-2)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_rows = 2 * torch.arange(N, device=device)
    sin_rows = 2 * torch.arange(N, device=device) + 1
    cols = torch.arange(N, device=device)
    A_data[..., cos_rows, cols] = cos_theta[..., :-1]
    A_data[..., cos_rows, cols + 1] = cos_theta[..., 1:]
    A_data[..., sin_rows, cols] = sin_theta[..., :-1]
    A_data[..., sin_rows, cols + 1] = sin_theta[..., 1:]
    Aw_data = A_data * torch.repeat_interleave(w, 2, dim=-1).unsqueeze(-1)
    ATA = einops.einsum(Aw_data, A_data, "... i j, ... i k -> ... j k")
    rhs = einops.einsum(Aw_data, b_data, "... i j, ... i -> ... j")
    DTD = construct_DTD(
        N + 1,
        lead,
        device=device,
        dtype=dtype,
        w_smooth3=1.0,
        lam=v_lambda,
        dt=dt,
    )
    lhs = ATA + DTD + v_ridge * torch.eye(N + 1, dtype=dtype, device=device).expand(*lead, N + 1, N + 1)
    L = torch.linalg.cholesky(lhs)
    return torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)


@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
@torch._dynamo.disable()
def dxy_theta_to_v(
    dxy: torch.Tensor,
    theta: torch.Tensor,
    v0: torch.Tensor,
    dt: float = 1.0,
    v_lambda: float = 1e-4,
    v_ridge: float = 1e-4,
) -> torch.Tensor:
    *lead, N, _ = dxy.shape
    device, dtype = dxy.device, dxy.dtype
    g = 2 / dt * dxy
    w = torch.ones_like(dxy[..., 0])
    A_data = torch.zeros(*lead, 2 * N, N + 1, dtype=dtype, device=device)
    b_data = g.flatten(start_dim=-2)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_rows = 2 * torch.arange(N, device=device)
    sin_rows = 2 * torch.arange(N, device=device) + 1
    cols = torch.arange(N, device=device)
    A_data[..., cos_rows, cols] = cos_theta[..., :-1]
    A_data[..., cos_rows, cols + 1] = cos_theta[..., 1:]
    A_data[..., sin_rows, cols] = sin_theta[..., :-1]
    A_data[..., sin_rows, cols + 1] = sin_theta[..., 1:]
    Aw_data = A_data * torch.repeat_interleave(w, 2, dim=-1).unsqueeze(-1)
    ATA = einops.einsum(Aw_data, A_data, "... i j, ... i k -> ... j k")
    rhs = einops.einsum(Aw_data[..., :, 1:], b_data, "... i j, ... i -> ... j")
    rhs -= ATA[..., 1:, 0] * v0.unsqueeze(-1)
    DTD = construct_DTD(
        N + 1,
        lead,
        device=device,
        dtype=dtype,
        w_smooth3=1.0,
        lam=v_lambda,
        dt=dt,
    )
    rhs -= DTD[..., 1:, 0] * v0.unsqueeze(-1)
    lhs = ATA[..., 1:, 1:] + DTD[..., 1:, 1:] + v_ridge * torch.eye(N, dtype=dtype, device=device).expand(
        *lead, N, N
    )
    L = torch.linalg.cholesky(lhs)
    y = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
    return torch.cat([v0.unsqueeze(-1), y], dim=-1)


@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
@torch._dynamo.disable()
def theta_smooth(
    traj_future_rot: torch.Tensor,
    dt: float = 1.0,
    theta_lambda: float = 1e-4,
    theta_ridge: float = 1e-4,
) -> torch.Tensor:
    theta = so3_to_yaw_torch(traj_future_rot)
    theta = unwrap_angle(theta)
    return solve_single_constraint(
        x_init=torch.zeros_like(theta[..., 0]),
        x_target=theta,
        w_smooth3=1.0,
        dt=dt,
        lam=theta_lambda,
        ridge=theta_ridge,
    )


class UnicycleAccelCurvatureActionSpace(ActionSpace):
    def __init__(
        self,
        accel_mean: float = 0.0,
        accel_std: float = 1.0,
        curvature_mean: float = 0.0,
        curvature_std: float = 1.0,
        accel_bounds: tuple[float, float] = (-9.8, 9.8),
        curvature_bounds: tuple[float, float] = (-0.2, 0.2),
        dt: float = 0.1,
        n_waypoints: int = 64,
        theta_lambda: float = 1e-6,
        theta_ridge: float = 1e-8,
        v_lambda: float = 1e-6,
        v_ridge: float = 1e-4,
        a_lambda: float = 1e-4,
        a_ridge: float = 1e-4,
        kappa_lambda: float = 1e-4,
        kappa_ridge: float = 1e-4,
    ):
        super().__init__()
        # self.register_buffer("accel_mean", torch.tensor(accel_mean), persistent=False)
        # self.register_buffer("accel_std", torch.tensor(accel_std), persistent=False)
        # self.register_buffer("curvature_mean", torch.tensor(curvature_mean), persistent=False)
        # self.register_buffer("curvature_std", torch.tensor(curvature_std), persistent=False)
        self.accel_mean = float(accel_mean)
        self.accel_std = float(accel_std)
        self.curvature_mean = float(curvature_mean)
        self.curvature_std = float(curvature_std)

        self.accel_bounds = accel_bounds
        self.curvature_bounds = curvature_bounds
        self.dt = dt
        self.n_waypoints = n_waypoints
        self.theta_lambda = theta_lambda
        self.theta_ridge = theta_ridge
        self.v_lambda = v_lambda
        self.v_ridge = v_ridge
        self.a_lambda = a_lambda
        self.a_ridge = a_ridge
        self.kappa_lambda = kappa_lambda
        self.kappa_ridge = kappa_ridge

    def get_action_space_dims(self) -> tuple[int, int]:
        return (self.n_waypoints, 2)

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def _v_to_a(self, v: torch.Tensor) -> torch.Tensor:
        dv = (v[..., 1:] - v[..., :-1]) / self.dt
        return solve_xs_eq_y(
            s=torch.ones_like(dv),
            y=dv,
            dt=self.dt,
            lam=self.a_lambda,
            ridge=self.a_ridge,
            w_smooth2=1.0,
        )

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def _theta_v_a_to_kappa(self, theta: torch.Tensor, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        dtheta = theta[..., 1:] - theta[..., :-1]
        s = self.dt * v[..., :-1] + (self.dt**2) / 2.0 * a
        return solve_xs_eq_y(
            s=s,
            y=dtheta,
            w_data=torch.ones_like(dtheta),
            w_smooth2=1.0,
            lam=self.kappa_lambda,
            ridge=self.kappa_ridge,
            dt=self.dt,
        )

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def estimate_t0_states(self, traj_history_xyz: torch.Tensor, traj_history_rot: torch.Tensor) -> dict[str, torch.Tensor]:
        full_xy = traj_history_xyz[..., :2]
        dxy = full_xy[..., 1:, :] - full_xy[..., :-1, :]
        theta = unwrap_angle(so3_to_yaw_torch(traj_history_rot))
        v = dxy_theta_to_v_without_v0(dxy=dxy, theta=theta, dt=self.dt, v_lambda=self.v_lambda, v_ridge=self.v_ridge)
        return {"v": v[..., -1]}

    @torch.no_grad()
    @torch._dynamo.disable()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def traj_to_action(
        self,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
        traj_future_xyz: torch.Tensor,
        traj_future_rot: torch.Tensor,
        t0_states: dict[str, torch.Tensor] | None = None,
        output_all_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if traj_future_xyz.shape[-2] != self.n_waypoints:
            raise ValueError(f"future trajectory must have length {self.n_waypoints} but got {traj_future_xyz.shape[-2]}")
        if t0_states is None:
            t0_states = self.estimate_t0_states(traj_history_xyz, traj_history_rot)
        full_xy = torch.cat([traj_history_xyz[..., -1:, :], traj_future_xyz], dim=-2)[..., :2]
        dxy = full_xy[..., 1:, :] - full_xy[..., :-1, :]
        theta = theta_smooth(traj_future_rot=traj_future_rot, dt=self.dt, theta_lambda=self.theta_lambda, theta_ridge=self.theta_ridge)
        v = dxy_theta_to_v(dxy=dxy, theta=theta, v0=t0_states["v"], dt=self.dt, v_lambda=self.v_lambda, v_ridge=self.v_ridge)
        accel = self._v_to_a(v)
        kappa = self._theta_v_a_to_kappa(theta, v, accel)
        accel_mean = torch.as_tensor(self.accel_mean, device=accel.device, dtype=accel.dtype)
        accel_std = torch.as_tensor(self.accel_std, device=accel.device, dtype=accel.dtype)
        kappa_mean = torch.as_tensor(self.curvature_mean, device=kappa.device, dtype=kappa.dtype)
        kappa_std = torch.as_tensor(self.curvature_std, device=kappa.device, dtype=kappa.dtype)
        accel = (accel - accel_mean) / accel_std
        kappa = (kappa - kappa_mean) / kappa_std
        action = torch.stack([accel, kappa], dim=-1)
        if not output_all_states:
            return action
        return action, torch.stack([v[:, :-1], accel, theta[:, :-1]], dim=-1)

    def action_to_traj(
        self,
        action: torch.Tensor,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
        t0_states: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        accel, kappa = action[..., 0], action[..., 1]
        accel_mean = torch.as_tensor(self.accel_mean, device=accel.device, dtype=accel.dtype)
        accel_std = torch.as_tensor(self.accel_std, device=accel.device, dtype=accel.dtype)
        kappa_mean = torch.as_tensor(self.curvature_mean, device=kappa.device, dtype=kappa.dtype)
        kappa_std = torch.as_tensor(self.curvature_std, device=kappa.device, dtype=kappa.dtype)
        accel = accel * accel_std + accel_mean
        kappa = kappa * kappa_std + kappa_mean
        if t0_states is None:
            t0_states = self.estimate_t0_states(traj_history_xyz, traj_history_rot)
        v0 = t0_states["v"]
        velocity = torch.cat([v0.unsqueeze(-1), v0.unsqueeze(-1) + torch.cumsum(accel * self.dt, dim=-1)], dim=-1)
        theta = torch.cat(
            [
                torch.zeros_like(v0).unsqueeze(-1),
                torch.cumsum(kappa * velocity[..., :-1] * self.dt, dim=-1)
                + torch.cumsum(kappa * accel * 0.5 * (self.dt**2), dim=-1),
            ],
            dim=-1,
        )
        half_dt = 0.5 * self.dt
        x = torch.cumsum(velocity[..., :-1] * torch.cos(theta[..., :-1]) * half_dt, dim=-1) + torch.cumsum(
            velocity[..., 1:] * torch.cos(theta[..., 1:]) * half_dt,
            dim=-1,
        )
        y = torch.cumsum(velocity[..., :-1] * torch.sin(theta[..., :-1]) * half_dt, dim=-1) + torch.cumsum(
            velocity[..., 1:] * torch.sin(theta[..., 1:]) * half_dt,
            dim=-1,
        )
        batch_dim = traj_history_xyz.shape[:-2]
        traj_future_xyz = torch.zeros(*batch_dim, self.n_waypoints, 3, device=traj_history_xyz.device, dtype=traj_history_xyz.dtype)
        traj_future_xyz[..., 0] = x
        traj_future_xyz[..., 1] = y
        traj_future_xyz[..., 2] = traj_history_xyz[..., -1:, 2]
        traj_future_rot = rot_2d_to_3d(rotation_matrix_torch(theta[..., 1:]))
        return traj_future_xyz, traj_future_rot


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.type_as(x) * self.weight


class MLPEncoder(nn.Module):
    def __init__(self, num_input_feats: int, num_enc_layers: int, hidden_size: int, outdim: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(num_input_feats, hidden_size), nn.SiLU()]
        for layer_index in range(num_enc_layers):
            if layer_index < num_enc_layers - 1:
                layers.extend([RMSNorm(hidden_size, eps=1e-5), nn.Linear(hidden_size, hidden_size), nn.SiLU()])
            else:
                layers.extend([RMSNorm(hidden_size, eps=1e-5), nn.Linear(hidden_size, outdim)])
        self.trunk = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


class FourierEncoderV2(nn.Module):
    def __init__(self, dim: int, max_freq: float = 100.0):
        super().__init__()
        self.out_dim = dim
        self.half = dim // 2
        self.max_freq = float(max_freq)

    def _build_freqs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logspace(
            0,
            math.log10(self.max_freq),
            steps=self.half,
            device=x.device,
            dtype=x.dtype,
        )[None, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freqs = self._build_freqs(x)
        arg = x[..., None] * freqs * 2 * torch.pi
        return torch.cat([torch.sin(arg), torch.cos(arg)], -1) * math.sqrt(2)


class PerWaypointActionInProjV2(nn.Module):
    def __init__(
        self,
        in_dims: list[int],
        out_dim: int,
        num_enc_layers: int = 4,
        hidden_size: int = 1024,
        max_freq: float = 100.0,
        num_fourier_feats: int = 20,
    ):
        super().__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        self.sinus = nn.ModuleList(
            [FourierEncoderV2(dim=num_fourier_feats, max_freq=max_freq) for _ in range(in_dims[-1])]
        )
        self.timestep_fourier_encoder = FourierEncoderV2(dim=num_fourier_feats, max_freq=max_freq)
        num_input_feats = sum(s.out_dim for s in self.sinus) + self.timestep_fourier_encoder.out_dim
        self.encoder = MLPEncoder(
            num_input_feats=num_input_feats,
            num_enc_layers=num_enc_layers,
            hidden_size=hidden_size,
            outdim=out_dim,
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        batch_size, num_steps, _ = x.shape
        action_feats = torch.cat([encoder(x[:, :, i]) for i, encoder in enumerate(self.sinus)], dim=-1)
        timestep_feats = self.timestep_fourier_encoder(timesteps[..., -1]).repeat(1, num_steps, 1)
        fused = torch.cat((action_feats, timestep_feats), dim=-1)
        return self.norm(self.encoder(fused.flatten(0, 1)).reshape(batch_size, num_steps, -1))


class FlowMatching(nn.Module):
    def __init__(
        self,
        x_dims: list[int] | tuple[int, ...] | int,
        use_classifier_free_guidance: bool = False,
        int_method: Literal["euler"] = "euler",
        num_inference_steps: int = 10,
        inference_guidance_weight: float = 1.0,
        *args: Any,
        **kwargs: Any,
    ):
        del args, kwargs
        super().__init__()
        self.x_dims = [x_dims] if isinstance(x_dims, int) else list(x_dims)
        self.use_classifier_free_guidance = use_classifier_free_guidance
        self.int_method = int_method
        self.num_inference_steps = num_inference_steps
        self.inference_guidance_weight = inference_guidance_weight

    @staticmethod
    def _guided_v(
        step_fn: StepFn,
        x: torch.Tensor,
        t: torch.Tensor,
        unguided_step_fn: StepFn,
        inference_guidance_weight: float,
    ) -> torch.Tensor:
        guided_v = step_fn(x=x, t=t)
        unguided_v = unguided_step_fn(x=x, t=t)
        return (1 - inference_guidance_weight) * unguided_v + inference_guidance_weight * guided_v

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        step_fn: StepFn,
        unguided_step_fn: StepFn | None = None,
        device: torch.device = torch.device("cpu"),
        return_all_steps: bool = False,
        inference_step: int | None = None,
        int_method: Literal["euler"] | None = None,
        use_classifier_free_guidance: bool | None = None,
        inference_guidance_weight: float | None = None,
        temperature: float = 1.0,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del args, kwargs
        int_method = int_method or self.int_method
        if int_method != "euler":
            raise ValueError(f"Invalid integration method: {int_method}")
        inference_step = inference_step or self.num_inference_steps
        if use_classifier_free_guidance is None:
            use_classifier_free_guidance = self.use_classifier_free_guidance
        if inference_guidance_weight is None:
            inference_guidance_weight = self.inference_guidance_weight
        if use_classifier_free_guidance and unguided_step_fn is None:
            raise ValueError("unguided_step_fn is required when using classifier free guidance")

        x = torch.randn(batch_size, *self.x_dims, device=device) * temperature
        time_steps = torch.linspace(0.0, 1.0, inference_step + 1, device=device)
        n_dim = len(self.x_dims)
        all_steps = [x] if return_all_steps else None
        for i in range(inference_step):
            dt = time_steps[i + 1] - time_steps[i]
            dt = dt.view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)
            t_start = time_steps[i].view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)
            if use_classifier_free_guidance:
                v = self._guided_v(
                    step_fn=step_fn,
                    x=x,
                    t=t_start,
                    unguided_step_fn=unguided_step_fn,
                    inference_guidance_weight=inference_guidance_weight,
                )
            else:
                v = step_fn(x=x, t=t_start)
            x = x + dt * v
            if return_all_steps:
                assert all_steps is not None
                all_steps.append(x)
        if return_all_steps:
            assert all_steps is not None
            return torch.stack(all_steps, dim=1), time_steps
        return x
