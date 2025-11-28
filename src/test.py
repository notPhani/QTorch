import math
import random
import torch
import time
from circuit import Circuit
from gates import GateOp
from backend import StatevectorBackend

def build_random_circuit(n_qubits, depth, seed=42):
    """
    Build a brick-wall random circuit:
    - Each layer alternates between even and odd pairs
    - Random single-qubit rotations between 2q layers
    """
    random.seed(seed)
    c = Circuit(n_qubits)
    
    for d in range(depth):
        # Layer 1: random single-qubit rotations on all qubits
        for q in range(n_qubits):
            gate = random.choice(["Rx", "Ry", "Rz"])
            angle = random.uniform(0, 2 * math.pi)
            c.add(GateOp(gate, [q], params=[angle]))
        
        # Layer 2: entangling layer (brick pattern)
        if d % 2 == 0:
            # even pairs: (0,1), (2,3), (4,5), ...
            pairs = [(i, i+1) for i in range(0, n_qubits-1, 2)]
        else:
            # odd pairs: (1,2), (3,4), (5,6), ...
            pairs = [(i, i+1) for i in range(1, n_qubits-1, 2)]
        
        for q1, q2 in pairs:
            # mix of CNOT and CZ for variety
            gate = random.choice(["CNOT", "CZ"])
            c.add(GateOp(gate, [q1, q2]))
    
    return c

c = build_random_circuit(n_qubits=2, depth=100)

# Count gates
total_gates = sum(len([g for g in row if g is not None]) for row in c.grid)
print(f"Total gate operations: {total_gates}")
print(f"Circuit depth: {max(len(row) for row in c.grid)}")

# Execute
backend = StatevectorBackend(2, device="cuda")
print("\nExecuting...")
start = time.time()
final_state = c.execute(backend)
elapsed = time.time() - start

print(f"Execution time: {elapsed:.2f} seconds")
print(f"Final state shape: {final_state.shape}")
print(f"State norm: {torch.linalg.norm(final_state).item():.10f}")
print(f"Nonzero amplitudes: {torch.count_nonzero(torch.abs(final_state) > 1e-10).item()}")

c = Circuit(3)
c.add(GateOp("H", [0]))
c.add(GateOp("CNOT", [0, 1]))
c.add(GateOp("CNOT", [1, 2]))

backend = StatevectorBackend(3)
c.execute(backend)
print(backend.state)

c = Circuit(2)
c.add(GateOp("H", [0]))
c.add(GateOp("CNOT", [0, 1]))
# now decode
c.add(GateOp("CNOT", [0, 1]))
c.add(GateOp("H", [0]))

backend = StatevectorBackend(2)
c.execute(backend)
print(backend.state)

c = Circuit(3)

# Prepare |ψ>
c.add(GateOp("H", [0]))
c.add(GateOp("P", [0], params={math.pi/2}))   # phase to create i

# Bell pair
c.add(GateOp("H", [1]))
c.add(GateOp("CNOT", [1, 2]))

# Entangle ψ with Alice's half
c.add(GateOp("CNOT", [0, 1]))
c.add(GateOp("H", [0]))

# Measure Alice
m0 = GateOp("M", [0])
m1 = GateOp("M", [1])
c.add(m0)
c.add(m1)

# Conditioned corrections
c.add(GateOp("Zc", [2], depends_on=[m0, m1]))
c.add(GateOp("Xc", [2], depends_on=[m0, m1]))

backend = StatevectorBackend(3)
c.execute(backend)
print(backend.state)

c = Circuit(3)

# Forward QFT
c.add(GateOp("H", [0]))
c.add(GateOp("CP", [1, 0], params={math.pi/2}))
c.add(GateOp("CP", [2, 0], params={math.pi/4}))

c.add(GateOp("H", [1]))
c.add(GateOp("CP", [2, 1], params={math.pi/2}))

c.add(GateOp("H", [2]))
c.add(GateOp("SWAP", [0, 2]))

# Inverse QFT (QFT†)
c.add(GateOp("SWAP", [0, 2]))
c.add(GateOp("H", [2]))
c.add(GateOp("CP†", [2, 1], params={-math.pi/2}))
c.add(GateOp("H", [1]))
c.add(GateOp("CP†", [2, 0], params={-math.pi/4}))
c.add(GateOp("CP†", [1, 0], params={-math.pi/2}))
c.add(GateOp("H", [0]))

backend = StatevectorBackend(3)
c.execute(backend)
print(backend.state)

import math

c = Circuit(3)

# Uniform superposition
for q in range(3):
    c.add(GateOp("H", [q]))

# Oracle that flips phase of |101>
c.add(GateOp("X", [0]))
c.add(GateOp("X", [2]))
c.add(GateOp("CCZ", [0,1,2]))   # or use decomposed version
c.add(GateOp("X", [0]))
c.add(GateOp("X", [2]))

# Diffusion
for q in range(3):
    c.add(GateOp("H", [q]))
for q in range(3):
    c.add(GateOp("X", [q]))

c.add(GateOp("CCZ", [0,1,2]))

for q in range(3):
    c.add(GateOp("X", [q]))
for q in range(3):
    c.add(GateOp("H", [q]))

backend = StatevectorBackend(3)
c.execute(backend)
print(backend.state)
