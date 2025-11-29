import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

st.set_page_config(page_title="SIRD Intervention Simulator", layout="wide")

st.title("SIRD Model with Policy Interventions")
st.markdown(
    """
This app simulates a **SIRD epidemic model** with a simple representation of
non-pharmaceutical interventions (NPIs), such as lockdowns or contact restrictions.

The transmission rate is reduced after a chosen *lockdown day*:

\[
\beta_{\text{eff}}(t) =
\begin{cases}
\beta, & t < t_{\text{lock}}, \\
(1-c)\,\beta, & t \ge t_{\text{lock}},
\end{cases}
\]

where:

- \(\beta\) is the baseline transmission rate,
- \(c \in [0,1]\) is the **contact reduction** (e.g. 0.4 = 40% reduction),
- \(t_{\text{lock}}\) is the day from which restrictions start to apply.
"""
)

# --- Sidebar: global settings ------------------------------------------
st.sidebar.header("Simulation settings")

T = st.sidebar.slider(
    "Simulation horizon (days)",
    min_value=60, max_value=365, value=365, step=5
)

# --- Population and initial conditions ---------------------------------
st.sidebar.subheader("Population & initial conditions")

N = st.sidebar.number_input(
    "Population size N",
    min_value=1_000_000, max_value=100_000_000,
    value=38_000_000, step=1_000_000
)

I0 = st.sidebar.number_input(
    "Initial infectious I(0)",
    min_value=1.0, max_value=float(N),
    value=1_000.0, step=100.0
)
R0 = st.sidebar.number_input(
    "Initial recovered R(0)",
    min_value=0.0, max_value=float(N),
    value=0.0, step=1_000.0
)
D0 = st.sidebar.number_input(
    "Initial deaths D(0)",
    min_value=0.0, max_value=float(N),
    value=0.0, step=1_000.0
)

S0 = max(N - I0 - R0 - D0, 0.0)

# --- Epidemiological parameters ----------------------------------------
st.sidebar.subheader("Epidemiological parameters")

beta = st.sidebar.number_input(
    "Transmission rate β",
    min_value=0.0, max_value=5.0,
    value=0.3, step=0.01, format="%.3f"
)
gamma = st.sidebar.number_input(
    "Recovery rate γ",
    min_value=0.0, max_value=1.0,
    value=0.2, step=0.01, format="%.3f"
)
mu = st.sidebar.number_input(
    "Mortality rate μ",
    min_value=0.0, max_value=0.1,
    value=0.001, step=0.001, format="%.3f"
)

# --- Policy parameters --------------------------------------------------
st.sidebar.subheader("Policy / behaviour")

contact_reduction = st.sidebar.slider(
    "Contact reduction c (0 = none, 0.5 = 50% less contacts)",
    min_value=0.0, max_value=0.9, value=0.3, step=0.05
)

lockdown_day = st.sidebar.slider(
    "Lockdown / intervention day t_lock",
    min_value=0, max_value=T-1, value=min(60, T // 3), step=5
)

# --- Basic reproduction numbers before/after intervention --------------
if gamma + mu > 0:
    R0_before = beta / (gamma + mu)
    R0_after = (beta * (1 - contact_reduction)) / (gamma + mu)
else:
    R0_before = float("nan")
    R0_after = float("nan")

st.markdown(
    f"""
**Implied basic reproduction numbers:**

- Before intervention: \( R_0^{{\text{{before}}}} \approx {R0_before:.2f} \)  
- After intervention:  \( R_0^{{\text{{after}}}} \approx {R0_after:.2f} \)
"""
)

# --- SIRD RHS with control ---------------------------------------------
def sird_rhs_control(t, y, beta_base, gamma, mu, N, contact_reduction, lockdown_day):
    S, I, R, D = y
    beta_eff = beta_base * (1.0 - contact_reduction) if t >= lockdown_day else beta_base
    dS = -beta_eff * S * I / N
    dI = beta_eff * S * I / N - gamma * I - mu * I
    dR = gamma * I
    dD = mu * I
    return [dS, dI, dR, dD]

# --- Time grid and initial state ---------------------------------------
t_eval = np.linspace(0, T, T + 1)
y0 = [S0, I0, R0, D0]

# --- Solve ODE ----------------------------------------------------------
sol = solve_ivp(
    fun=lambda t, y: sird_rhs_control(
        t, y, beta, gamma, mu, N,
        contact_reduction, lockdown_day
    ),
    t_span=(0, T),
    y0=y0,
    t_eval=t_eval,
    rtol=1e-6,
    atol=1e-6
)

S_sim, I_sim, R_sim, D_sim = sol.y

# --- Plot: 2x2 grid for S, I, R, D -------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
axS, axI, axR, axD = axes.ravel()

axS.plot(t_eval, S_sim)
axS.set_title("Susceptible S(t)")
axS.set_ylabel("Individuals")

axI.plot(t_eval, I_sim)
axI.set_title("Infectious I(t)")

axR.plot(t_eval, R_sim)
axR.set_title("Recovered R(t)")
axR.set_ylabel("Individuals")
axR.set_xlabel("Time t (days)")

axD.plot(t_eval, D_sim)
axD.set_title("Deaths D(t)")
axD.set_xlabel("Time t (days)")

for ax in axes.ravel():
    ax.axvline(x=lockdown_day, color="gray", linestyle="--", linewidth=1)
    ax.grid(alpha=0.3)

fig.suptitle(
    f"SIRD dynamics with contact reduction c = {contact_reduction:.2f} "
    f"from day t = {lockdown_day}",
    fontsize=13
)
plt.tight_layout()
st.pyplot(fig)

st.markdown(
    """
### How to use this app

1. **Set epidemiological parameters** (β, γ, μ) in the sidebar.
2. **Choose the intervention strength** `c` and **its start day** `t_lock`.
3. Observe how the four trajectories:
   - Susceptible \(S(t)\),
   - Infectious \(I(t)\),
   - Recovered \(R(t)\),
   - Deaths \(D(t)\)

   change in response to stronger / weaker and earlier / later restrictions.

The vertical dashed line indicates the day when the intervention starts.
"""
)
