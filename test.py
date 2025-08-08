import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def wrap_angle(x):
    """Wrap angle to [-pi, pi)."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def double_pendulum_rhs(t, y, p):
    """
    Right-hand side of double pendulum ODEs.
    y = [theta1, omega1, theta2, omega2]
    """
    theta1, omega1, theta2, omega2 = y
    m1, m2, L1, L2, g = p["m1"], p["m2"], p["L1"], p["L2"], p["g"]
    Delta = theta1 - theta2

    cosD = np.cos(Delta)
    sinD = np.sin(Delta)

    # Denominator from configuration-dependent inertia
    D = m1 + m2 - m2 * cosD**2

    # Avoid pathological division (should not happen for physical params)
    if np.any(np.isclose(D, 0.0)):
        D = D + 1e-12

    # Explicit accelerations (theta_ddot)
    num1 = (
        -g * (m1 + m2) * np.sin(theta1)
        + m2 * g * np.sin(theta2) * cosD
        - m2 * sinD * (L2 * omega2**2 + L1 * omega1**2 * cosD)
    )
    alpha1 = num1 / (L1 * D)

    num2 = (
        (m1 + m2) * g * (np.sin(theta1) * cosD - np.sin(theta2))
        + sinD * ((m1 + m2) * L1 * omega1**2 + m2 * L2 * omega2**2 * cosD)
    )
    alpha2 = num2 / (L2 * D)

    return np.array([omega1, alpha1, omega2, alpha2])


def total_energy(y, p):
    """Compute total mechanical energy E=T+V for the double pendulum."""
    theta1, omega1, theta2, omega2 = y
    m1, m2, L1, L2, g = p["m1"], p["m2"], p["L1"], p["L2"], p["g"]
    Delta = theta1 - theta2

    # Kinetic energy
    T = (
        0.5 * (m1 + m2) * L1**2 * omega1**2
        + 0.5 * m2 * L2**2 * omega2**2
        + m2 * L1 * L2 * omega1 * omega2 * np.cos(Delta)
    )
    # Potential energy (zero at pivot height)
    V = -(m1 + m2) * g * L1 * np.cos(theta1) - m2 * g * L2 * np.cos(theta2)
    return T + V


def simulate(p, y0, t_span=(0.0, 30.0), samples=5000, rtol=1e-9, atol=1e-9, max_step=None):
    """
    Integrate the double pendulum ODE.
    Returns t, sol (N x 4), energy (N,)
    """
    t_eval = np.linspace(t_span[0], t_span[1], samples)

    rhs = lambda t, y: double_pendulum_rhs(t, y, p)
    sol_ivp = solve_ivp(
        rhs,
        t_span,
        y0,
        t_eval=t_eval,
        method="DOP853",  # high-order explicit RK
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        vectorized=False,
    )

    if not sol_ivp.success:
        print("Integration warning:", sol_ivp.message)

    t = sol_ivp.t
    sol = sol_ivp.y.T  # shape (N, 4)

    # Energy monitoring
    E = np.array([total_energy(state, p) for state in sol])

    return t, sol, E


def plot_results(t, sol, E, title_suffix=""):
    theta1 = sol[:, 0]
    omega1 = sol[:, 1]
    theta2 = sol[:, 2]
    omega2 = sol[:, 3]

    # Wrap angles for phase portraits
    th1_wrapped = wrap_angle(theta1)
    th2_wrapped = wrap_angle(theta2)

    E0 = E[0]
    dE = E - E0
    rel_dE = dE / (abs(E0) + 1e-16)

    # Print energy drift summary
    print(f"Initial energy E0 = {E0:.8f}")
    print(f"Final   energy EN = {E[-1]:.8f}")
    print(f"Max |ΔE|/|E0| over trajectory = {np.max(np.abs(rel_dE)):.3e}")

    # Figure 1: Time series and energy drift
    fig1, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig1.suptitle(f"Double Pendulum Dynamics {title_suffix}")

    axs[0, 0].plot(t, th1_wrapped, label=r"$\theta_1$")
    axs[0, 0].plot(t, th2_wrapped, label=r"$\theta_2$", alpha=0.8)
    axs[0, 0].set_xlabel("t [s]")
    axs[0, 0].set_ylabel("angle [rad] (wrapped)")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(t, omega1, label=r"$\dot{\theta}_1$")
    axs[0, 1].plot(t, omega2, label=r"$\dot{\theta}_2$", alpha=0.8)
    axs[0, 1].set_xlabel("t [s]")
    axs[0, 1].set_ylabel("angular velocity [rad/s]")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    axs[1, 0].plot(t, dE, color="tab:red")
    axs[1, 0].set_xlabel("t [s]")
    axs[1, 0].set_ylabel("ΔE [J]")
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].set_title("Energy drift (absolute)")

    axs[1, 1].plot(t, rel_dE, color="tab:orange")
    axs[1, 1].set_xlabel("t [s]")
    axs[1, 1].set_ylabel("ΔE / |E0|")
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].set_title("Energy drift (relative)")

    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Figure 2: Phase portraits
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle(f"Phase Portraits {title_suffix}")

    axs2[0].plot(th1_wrapped, omega1, lw=0.8)
    axs2[0].set_xlabel(r"$\theta_1$ [rad] (wrapped)")
    axs2[0].set_ylabel(r"$\dot{\theta}_1$ [rad/s]")
    axs2[0].grid(True, alpha=0.3)
    axs2[0].set_title("(θ1, ω1)")

    axs2[1].plot(th2_wrapped, omega2, lw=0.8, color="tab:green")
    axs2[1].set_xlabel(r"$\theta_2$ [rad] (wrapped)")
    axs2[1].set_ylabel(r"$\dot{\theta}_2$ [rad/s]")
    axs2[1].grid(True, alpha=0.3)
    axs2[1].set_title("(θ2, ω2)")

    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()


if __name__ == "__main__":
    # Parameters (SI units)
    params = {
        "m1": 1.0,  # kg
        "m2": 1.0,  # kg
        "L1": 1.0,  # m
        "L2": 1.0,  # m
        "g": 9.81,  # m/s^2
    }

    # Initial conditions: [theta1, omega1, theta2, omega2]
    # Example: moderately energetic to see complex motion
    deg = np.deg2rad
    y0 = np.array([deg(120.0), 0.0, deg(-10.0), 0.0])

    # Simulation span
    t_span = (0.0, 30.0)  # seconds
    samples = 6000
    rtol = 1e-10
    atol = 1e-10
    max_step = 0.02  # limit step to improve energy behavior

    t, sol, E = simulate(params, y0, t_span=t_span, samples=samples, rtol=rtol, atol=atol, max_step=max_step)
    plot_results(t, sol, E, title_suffix=f"(m1=m2=1, L1=L2=1)")