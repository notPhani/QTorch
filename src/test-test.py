import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. Basic utilities
# -------------------------------------------------

def softmax(phi):
    """Convert raw φ values into valid probabilities."""
    e = np.exp(phi - np.max(phi))
    return e / e.sum()

# Pauli matrices
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
paulis = [I, X, Y, Z]

def apply_pauli_channel(rho, p):
    """Applies a single-qubit Pauli CPTP channel."""
    out = np.zeros_like(rho, dtype=complex)
    for weight, P in zip(p, paulis):
        out += weight * P @ rho @ P
    return out

def fidelity(rho, sigma):
    """State fidelity for pure or mixed density matrices."""
    sqrt_rho = scipy.linalg.sqrtm(rho)
    inner = scipy.linalg.sqrtm(sqrt_rho @ sigma @ sqrt_rho)
    return np.real(np.trace(inner))**2


# -------------------------------------------------
# 2. φ update rule
# -------------------------------------------------

def update_phi(phi, alpha, lam, phi_eq, beta, W,
               G, M, rho_coeff, H, sigma):
    noise = np.random.normal(size=phi.shape)
    return ((alpha - lam) * phi
            + lam * phi_eq
            + beta * W @ phi
            + G
            + M
            + rho_coeff * H(phi)
            + sigma * noise)


# -------------------------------------------------
# 3. Simulation driver
# -------------------------------------------------

def simulate(num_steps, rho0, phi0, params):
    rho = rho0.copy()
    phi = phi0.copy()

    rho_list = []
    phi_list = []
    p_list = []

    for t in range(num_steps):

        # Update φ
        phi = update_phi(phi, **params)

        # Softmax → Pauli probabilities
        p = softmax(phi)

        # Apply the quantum channel
        rho = apply_pauli_channel(rho, p)

        rho_list.append(rho.copy())
        phi_list.append(phi.copy())
        p_list.append(p.copy())

    return np.array(rho_list), np.array(phi_list), np.array(p_list)


# -------------------------------------------------
# 4. Plotting tools
# -------------------------------------------------

def plot_phi(phi_list):
    plt.figure(figsize=(7, 4))
    plt.plot(phi_list)
    plt.title("φ Drift Over Time")
    plt.xlabel("Time step")
    plt.ylabel("φ values")
    plt.grid()
    plt.show()

def plot_probabilities(p_list):
    plt.figure(figsize=(7, 4))
    for i, label in enumerate(['I', 'X', 'Y', 'Z']):
        plt.plot(p_list[:, i], label=label)
    plt.title("Pauli Error Probabilities Over Time")
    plt.xlabel("Time step")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid()
    plt.show()

def plot_error_spectrum(p_list):
    avg_p = p_list.mean(axis=0)
    plt.figure(figsize=(5, 5))
    plt.bar(['I', 'X', 'Y', 'Z'], avg_p)
    plt.title("Average Pauli Error Spectrum")
    plt.ylabel("Average probability")
    plt.grid(axis='y')
    plt.show()

def plot_autocorrelation(phi_list):
    plt.figure(figsize=(7, 4))
    for i in range(phi_list.shape[1]):
        x = phi_list[:, i] - np.mean(phi_list[:, i])
        corr = np.correlate(x, x, mode='full')
        corr = corr[corr.size//2:]
        corr /= corr[0]
        plt.plot(corr, label=f"φ[{i}]")
    plt.title("Noise Autocorrelation")
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    plt.grid()
    plt.legend()
    plt.show()

def plot_fidelity(rho_list, rho_ideal):
    import scipy.linalg
    F = []
    for rho in rho_list:
        sqrt_r = scipy.linalg.sqrtm(rho)
        tmp = scipy.linalg.sqrtm(sqrt_r @ rho_ideal @ sqrt_r)
        F.append(np.real(np.trace(tmp))**2)
    plt.figure(figsize=(7,4))
    plt.plot(F)
    plt.title("Fidelity vs Time")
    plt.xlabel("Time")
    plt.ylabel("Fidelity")
    plt.grid()
    plt.show()


# -------------------------------------------------
# 5. Quick test on import
# -------------------------------------------------

if __name__ == "__main__":
    import scipy.linalg

    # initial pure state |0>
    rho0 = np.array([[1, 0],
                     [0, 0]], dtype=complex)

    # φ controls p(I), p(X), p(Y), p(Z)
    phi0 = np.array([0.1, 0.2, -0.3, -0.1])

    # Parameters
    params = dict(
        alpha=0.92,
        lam=0.05,
        phi_eq=np.array([0.0, -1.0, -1.0, -1.0]),
        beta=0.01,
        W=np.eye(4),
        G=np.zeros(4),
        M=np.zeros(4),
        rho_coeff=0.01,
        H=lambda x: np.tanh(x),
        sigma=0.05
    )

    steps = 50

    rhos, phis, probs = simulate(steps, rho0, phi0, params)

    # Plot everything
    plot_phi(phis)
    plot_probabilities(probs)
    plot_error_spectrum(probs)
    plot_autocorrelation(phis)
    plot_fidelity(rhos, rho0)
