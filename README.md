# Neural-ODEs-on-DE
#The project uses neural ODEs to learn and estimate dynamics of differential equations

# Neural ODE for: y'' + y = x,  y(0)=1, y'(0)=1
# Learns the dynamics z' = f_theta(t, z), z = [y, v], with optional physics residual.

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Optional: torchdiffeq (if installed, training is a bit faster/stabler)
try:
    from torchdiffeq import odeint
    HAS_TORCHDIFFEQ = True
except Exception:
    HAS_TORCHDIFFEQ = False

device = torch.device("cpu")

# ----- Ground-truth (for supervised training data or evaluation) -----
def y_true_torch(t):   # y(x) = cos x + x
    return torch.cos(t) + t
def v_true_torch(t):   # y'(x) = -sin x + 1
    return -torch.sin(t) + 1

# ----- Neural ODE dynamics: f_theta(t, z) -> [y', v'] -----
class ODEFunc(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2)
        )
        # Init small-ish for stability
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.9)
                nn.init.zeros_(m.bias)

    def forward(self, t, z):
        # Supports scalar t with z shape (2,) OR batched t (N,) with z (N,2)
        if z.dim() == 1:  # single state
            inp = torch.cat([t.view(1), z], dim=0).view(1, -1)  # (1,3)
            out = self.net(inp)[0]  # (2,)
        else:              # batch
            inp = torch.cat([t[:, None], z], dim=1)  # (N,3)
            out = self.net(inp)                      # (N,2)
        return out

func = ODEFunc().to(device)

# ----- RK4 fallback integrator (autograd-friendly) -----
def rk4_step(func, t, z, dt):
    k1 = func(t, z)
    k2 = func(t + 0.5*dt, z + 0.5*dt*k1)
    k3 = func(t + 0.5*dt, z + 0.5*dt*k2)
    k4 = func(t + dt,     z + dt*k3)
    return z + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def integrate(func, z0, t):
    if HAS_TORCHDIFFEQ:
        return odeint(func, z0, t, method='dopri5')  # (T, 2)
    # Fallback: explicit RK4 unroll (keeps gradients)
    z = z0
    traj = [z0]
    for i in range(t.numel() - 1):
        dt = t[i+1] - t[i]
        z = rk4_step(func, t[i], z, dt)
        traj.append(z)
    return torch.stack(traj, dim=0)

# ----- Training setup -----
T_end = 10.0
N_train = 200
t_train = torch.linspace(0.0, T_end, N_train, device=device, dtype=torch.float32)

# Initial condition
z0 = torch.tensor([1.0, 1.0], device=device)  # [y(0), v(0)] = [1, 1]

# Targets (supervised); replace with your measured data if desired
y_target = y_true_torch(t_train)
v_target = v_true_torch(t_train)

optimizer = torch.optim.Adam(func.parameters(), lr=1e-3)
mse = nn.MSELoss()

# Loss weights
w_y = 1.0       # match y(t)
w_v = 0.5       # optionally match v(t) = y'(t)
w_phys = 0.2    # physics residual on v' + y - t

for epoch in range(2000):
    optimizer.zero_grad()

    # Integrate the neural ODE from the IC
    traj = integrate(func, z0, t_train)   # shape (N, 2); columns: [y, v]
    y_pred, v_pred = traj[:, 0], traj[:, 1]

    # Supervised losses to nudge toward the true solution (use your data here)
    loss_y = mse(y_pred, y_target)
    loss_v = mse(v_pred, v_target)

    # Physics residual: r(t) = v'(t) + y(t) - t  should be ~ 0
    # We can compute v' directly from the network: f_theta(t, z)_2
    with torch.set_grad_enabled(True):
        dz = func(t_train, traj)          # (N, 2)
    resid = dz[:, 1] + y_pred - t_train   # v' + y - t
    loss_phys = mse(resid, torch.zeros_like(resid))

    loss = w_y*loss_y + w_v*loss_v + w_phys*loss_phys
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"epoch {epoch+1:4d} | "
              f"Ly={loss_y.item():.3e} Lv={loss_v.item():.3e} Lphys={loss_phys.item():.3e} "
              f"Total={loss.item():.3e}")

# ----- Evaluation & plots -----
t_plot = torch.linspace(0.0, T_end, 400, device=device)
traj_plot = integrate(func, z0, t_plot).detach().cpu().numpy()
y_pred_plot, v_pred_plot = traj_plot[:, 0], traj_plot[:, 1]

x_np = t_plot.detach().cpu().numpy()
y_true_np = (np.cos(x_np) + x_np)
v_true_np = (-np.sin(x_np) + 1.0)

plt.figure()
plt.plot(x_np, y_true_np, linestyle="--", label="Exact y(x)")
plt.plot(x_np, y_pred_plot, label="Neural ODE y(x)")
plt.xlabel("x"); plt.ylabel("y")
plt.title("Neural ODE vs Exact: y'' + y = x")
plt.legend(); plt.grid(True)

plt.figure()
plt.plot(x_np, v_true_np, linestyle="--", label="Exact y'(x)")
plt.plot(x_np, v_pred_plot, label="Neural ODE y'(x)")
plt.xlabel("x"); plt.ylabel("y'")
plt.title("Velocity comparison")
plt.legend(); plt.grid(True)

plt.show()
