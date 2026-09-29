"""Physics-Informed Neural Network (PINN) for the Richardson-Richards
equation -- the fluid-dynamics piece of the multiphysics idea we scoped.

SCOPE (deliberately narrow, per the staged plan):
    - Moisture (theta) only. No chemistry/biology equations yet.
    - ONE soil column (fixed hydraulic parameters), not yet conditioned on
      your live satellite/climate features. That's the next step, once
      this reproduces a known result correctly.
    - Trained on a standard literature test case: infiltration into an
      initially dry sandy-loam column, top boundary held near-saturated.
      This is the same style of test problem used in Bandai & Ghezzehei
      (2021, 2022) and the Ireson et al. OpenRE reference solver -- so you
      can sanity-check this against their published figures/code, not
      just trust it blindly.

WHAT THIS DOES NOT DO YET:
    - Does not use your data/agrishield_training.csv at all -- there is no
      depth-resolved moisture data in it (recall: your WoSIS/LUCAS rows
      are filtered to upper_depth == 0, surface only). Wiring in real data
      is a later step once you have sensor readings (e.g. your ESP32 rig)
      or a public depth-resolved dataset to compare against.
    - Does not source K(theta)/van Genuchten parameters from HiHydroSoil
      per-coordinate -- they're hardcoded below for a known reference
      soil, on purpose, so you have a ground-truth-comparable baseline.

Run this file directly to train and see a plot of the result:
    python agrishield/pinn.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 0. Device -- use your RTX 5070 Ti if torch can see it, else fall back to CPU
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# 1. Soil hydraulic properties (van Genuchten-Mualem model)
#
# These are the K(theta) and psi(theta) constitutive relationships Richards'
# equation needs. Hardcoded here to a standard reference sandy loam so this
# script has a known, checkable answer. In a later, real version, these
# would be looked up per-coordinate from HiHydroSoil/GSHP instead of fixed.
# ---------------------------------------------------------------------------
THETA_R = 0.065   # residual water content
THETA_S = 0.41    # saturated water content
ALPHA = 7.5       # van Genuchten alpha, 1/m
N_VG = 1.89       # van Genuchten n (shape parameter)
M_VG = 1.0 - 1.0 / N_VG
K_SAT = 1.5       # saturated hydraulic conductivity, m/day


def effective_saturation(theta: torch.Tensor) -> torch.Tensor:
    """Se = (theta - theta_r) / (theta_s - theta_r), clamped to a safe
    range -- the van Genuchten formulas blow up exactly at Se=0 or Se=1,
    and the network's raw output will wander outside [theta_r, theta_s]
    early in training before it's learned to stay physical."""
    se = (theta - THETA_R) / (THETA_S - THETA_R)
    return torch.clamp(se, min=1e-4, max=1.0 - 1e-4)


def water_potential(theta: torch.Tensor) -> torch.Tensor:
    """psi(theta) -- capillary suction head, van Genuchten closed form.
    Negative by convention (suction), in metres."""
    se = effective_saturation(theta)
    return -(1.0 / ALPHA) * (se ** (-1.0 / M_VG) - 1.0) ** (1.0 / N_VG)


def hydraulic_conductivity(theta: torch.Tensor) -> torch.Tensor:
    """K(theta) -- Mualem-van Genuchten unsaturated conductivity model."""
    se = effective_saturation(theta)
    inner = 1.0 - (1.0 - se ** (1.0 / M_VG)) ** M_VG
    return K_SAT * torch.sqrt(se) * inner ** 2


# ---------------------------------------------------------------------------
# 2. The network
#
# Takes normalised depth z and time t, outputs predicted theta(z, t).
# This is an ordinary MLP -- nothing PINN-specific is in the architecture
# itself. The physics enters entirely through the LOSS function in step 4.
# ---------------------------------------------------------------------------
class RichardsPINN(nn.Module):
    def __init__(self, hidden_size: int = 64, n_hidden_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(2, hidden_size), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

        # Output squashed into [theta_r, theta_s] via sigmoid, so the
        # network can never predict a physically impossible moisture
        # value, however badly it's currently trained.
        self._range = THETA_S - THETA_R

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, t], dim=1)
        raw = self.net(x)
        return THETA_R + self._range * torch.sigmoid(raw)


# ---------------------------------------------------------------------------
# 3. Problem setup -- domain, initial condition, boundary conditions
#
# Standard test case: a 1-metre soil column, initially fairly dry
# throughout, with the top suddenly held near-saturated (ponded
# infiltration) for 1 day. This is the same class of problem used to
# validate the published Richards-equation PINNs, so it's a fair sanity
# check of this implementation.
# ---------------------------------------------------------------------------
Z_MAX = 1.0   # metres, column depth
T_MAX = 1.0   # days, simulation horizon

THETA_INITIAL = 0.10   # initially fairly dry
THETA_TOP_BC = 0.38    # near-saturated at the surface once infiltration starts


def sample_collocation_points(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Random interior (z, t) points -- no label needed here. This is
    where the physics loss gets enforced."""
    z = torch.rand(n, 1, device=device) * Z_MAX
    t = torch.rand(n, 1, device=device) * T_MAX
    z.requires_grad_(True)
    t.requires_grad_(True)
    return z, t


def sample_initial_points(n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = torch.rand(n, 1, device=device) * Z_MAX
    t = torch.zeros(n, 1, device=device)
    theta_target = torch.full((n, 1), THETA_INITIAL, device=device)
    return z, t, theta_target


def sample_top_boundary_points(n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = torch.zeros(n, 1, device=device)
    t = torch.rand(n, 1, device=device) * T_MAX
    theta_target = torch.full((n, 1), THETA_TOP_BC, device=device)
    return z, t, theta_target


# ---------------------------------------------------------------------------
# 4. The physics loss -- this is the actual PINN mechanism
#
# For a batch of (z, t) points, compute the network's own derivatives via
# autograd, plug them into the Richardson-Richards equation, and penalise
# whatever residual is left. If theta(z,t) truly satisfies the equation,
# this residual is exactly zero everywhere.
#
#     d(theta)/dt = d/dz [ K(theta) * (d(psi)/dz + 1) ]
# ---------------------------------------------------------------------------
def richards_residual(model: RichardsPINN, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    theta = model(z, t)

    # d(theta)/dt
    dtheta_dt = torch.autograd.grad(
        theta, t, grad_outputs=torch.ones_like(theta), create_graph=True
    )[0]

    # psi(theta) and its derivative w.r.t. z (chain rule handled by autograd
    # automatically, since psi is itself differentiable in terms of theta,
    # and theta is a function of z)
    psi = water_potential(theta)
    dpsi_dz = torch.autograd.grad(
        psi, z, grad_outputs=torch.ones_like(psi), create_graph=True
    )[0]

    # Darcy flux term: K(theta) * (d(psi)/dz + 1)   [the "+1" is gravity]
    K = hydraulic_conductivity(theta)
    flux = K * (dpsi_dz - 1.0)

    # d(flux)/dz -- second derivative overall, needs create_graph=True above
    dflux_dz = torch.autograd.grad(
        flux, z, grad_outputs=torch.ones_like(flux), create_graph=True
    )[0]

    return dtheta_dt - dflux_dz  # should be ~0 if physics is satisfied


# ---------------------------------------------------------------------------
# 5. Training loop -- combines physics loss + initial/boundary condition loss
#
# There's no "real data" loss term yet (no depth-labelled sensor readings
# to fit), so this version is trained purely on physics + IC/BC -- which is
# itself a legitimate, standard PINN use case (forward-solving a known PDE
# given known conditions, without needing interior labels at all). Once you
# have real readings (ESP32 rig, or a public dataset), add a data-loss term
# here the same way -- MSE against those readings, added to the total.
# ---------------------------------------------------------------------------
def train(n_epochs: int = 5000, n_collocation: int = 2000, n_ic: int = 200, n_bc: int = 200):
    model = RichardsPINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # physics loss
        z_f, t_f = sample_collocation_points(n_collocation)
        residual = richards_residual(model, z_f, t_f)
        physics_loss = torch.mean(residual ** 2)

        # initial condition loss (t=0 everywhere in the column)
        z_ic, t_ic, theta_ic_target = sample_initial_points(n_ic)
        theta_ic_pred = model(z_ic, t_ic)
        ic_loss = torch.mean((theta_ic_pred - theta_ic_target) ** 2)

        # top boundary condition loss (z=0, all t -- surface held wet)
        z_bc, t_bc, theta_bc_target = sample_top_boundary_points(n_bc)
        theta_bc_pred = model(z_bc, t_bc)
        bc_loss = torch.mean((theta_bc_pred - theta_bc_target) ** 2)

        # Weighting matters a lot for PINN training stability (this is the
        # documented failure mode we flagged earlier). IC/BC terms are
        # up-weighted here because with equal weighting the network tends
        # to satisfy the physics loss cheaply by ignoring the boundary
        # conditions -- a known, common PINN training pathology.
        loss = physics_loss + 50.0 * ic_loss + 50.0 * bc_loss

        loss.backward()
        optimizer.step()

        if epoch % 500 == 0 or epoch == n_epochs - 1:
            print(
                f"epoch {epoch:5d}  total {loss.item():.6f}  "
                f"physics {physics_loss.item():.6f}  "
                f"ic {ic_loss.item():.6f}  bc {bc_loss.item():.6f}"
            )

    return model


# ---------------------------------------------------------------------------
# 6. Run it, and plot the result -- moisture vs depth, at a few time slices.
# Sanity check: the wetting front should visibly move DOWN through the
# column as t increases, and the top should stay near THETA_TOP_BC while
# depth stays near THETA_INITIAL early on. If you see something that
# violates that basic physical picture, something's wrong before you trust
# any number out of this.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = train()

    torch.save(model.state_dict(), "pinn_moisture_reference_column.pt")
    print("saved model weights to pinn_moisture_reference_column.pt")

    try:
        import matplotlib.pyplot as plt

        z_plot = torch.linspace(0, Z_MAX, 200, device=device).reshape(-1, 1)
        plt.figure(figsize=(6, 5))
        for t_val in [0.0, 0.1, 0.3, 0.6, 1.0]:
            t_plot = torch.full_like(z_plot, t_val)
            with torch.no_grad():
                theta_pred = model(z_plot, t_plot).cpu().numpy()
            plt.plot(theta_pred, -z_plot.cpu().numpy(), label=f"t = {t_val:.1f} day")
        plt.xlabel("theta (volumetric water content)")
        plt.ylabel("depth (m)")
        plt.title("PINN solution: Richards' equation, infiltration test case")
        plt.legend()
        plt.tight_layout()
        plt.savefig("pinn_infiltration_result.png", dpi=150)
        print("saved plot to pinn_infiltration_result.png")
    except ImportError:
        print("matplotlib not installed -- skipping plot, weights are saved though")