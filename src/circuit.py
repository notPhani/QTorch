class GateGrid:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.grid = [[] for _ in range(num_qubits)]

    def _ensure(self, q, t):
        while len(self.grid[q]) <= t:
            self.grid[q].append(None)

    def add(self, gate):
        qubits = gate.qubits

        if gate.t is not None:
            t = gate.t
            for q in qubits:
                self._ensure(q, t)
                if self.grid[q][t] is not None:
                    raise ValueError(f"Qubit {q} busy at t={t}")
        else:
            last = max(len(self.grid[q]) - 1 for q in qubits) if qubits else -1
            t = last + 1
            while True:
                for q in qubits:
                    self._ensure(q, t)
                if any(self.grid[q][t] is not None for q in qubits):
                    t += 1
                    continue
                break
            gate.t = t

        for q in qubits:
            self.grid[q][t] = gate

        return t

class GateOp:
    def __init__(self, name, qubits, t=None):
        self.name = name
        self.qubits = qubits
        self.t = t
    def __repr__(self):
        return self.name

# Build teleportation circuit on 3 qubits: 0 (ψ), 1 (Alice), 2 (Bob)
circ = GateGrid(3)

# 1) Prepare Bell pair between qubits 1 and 2
circ.add(GateOp("H",   [1]))        # H on qubit 1
circ.add(GateOp("CNOT",[1, 2]))     # CNOT 1->2

# 2) Entangle data qubit 0 with Alice's Bell half (q1)
circ.add(GateOp("CNOT",[0, 1]))     # CNOT 0->1
circ.add(GateOp("H",   [0]))        # H on qubit 0

# 3) Measure Alice's qubits (0 and 1)
circ.add(GateOp("M",   [0]))
circ.add(GateOp("M",   [1]))

# 4) Classically controlled corrections on Bob's qubit
# (we just place them structurally; classical control is separate)
circ.add(GateOp("Z*",  [2]))        # Z correction placeholder
circ.add(GateOp("X*",  [2]))        # X correction placeholder

for row in circ.grid:
    print(row)
