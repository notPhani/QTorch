import numpy as np
import matplotlib.pyplot as plt

# Mock teleportation phi dynamics over 10 timesteps, 3 qubits
timesteps = 10
qubits = 3

# Laplacian-style: Local spread, structured propagation
phi_lap = np.zeros((qubits, timesteps))
phi_lap[:,0] = [1.0, 0.0, 0.0]  # spike at Q0

for t in range(1, timesteps):
    phi_lap[0,t] = phi_lap[0,t-1] * 0.8
    phi_lap[1,t] = phi_lap[0,t-1] * 0.4 + phi_lap[1,t-1] * 0.7
    phi_lap[2,t] = phi_lap[1,t-1] * 0.3 + phi_lap[2,t-1] * 0.7

# Exponential-style: smooth global leakage
phi_exp = np.zeros((qubits, timesteps))
phi_exp[:,0] = [1.0, 0.0, 0.0]

for t in range(1, timesteps):
    total = phi_exp[:,t-1]
    phi_exp[0,t] = total[0]*0.75 + total[1]*0.15 + total[2]*0.05
    phi_exp[1,t] = total[1]*0.75 + total[0]*0.15 + total[2]*0.10
    phi_exp[2,t] = total[2]*0.75 + total[1]*0.15 + total[0]*0.05

# Plot heatmaps
fig, axs = plt.subplots(1, 2, figsize=(10,4))

axs[0].imshow(phi_lap, aspect='auto')
axs[0].set_title("Laplacian Diffusion (Structured Transfer)")
axs[0].set_xlabel("Timesteps")
axs[0].set_ylabel("Qubit Index")

axs[1].imshow(phi_exp, aspect='auto')
axs[1].set_title("Exponential Fall-off (Global Leakage)")
axs[1].set_xlabel("Timesteps")
axs[1].set_ylabel("Qubit Index")

plt.tight_layout()
plt.show()
