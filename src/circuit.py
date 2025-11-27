class Ciruit:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.grid = [[] for _ in range(num_qubits)]
        # (name, qubit_digits) -> count
        self._label_counts = {}

    def _ensure(self, q, t):
        while len(self.grid[q]) <= t:
            self.grid[q].append(None)

    def _assign_label(self, gate):
        # qubits in the exact order given → direction preserved
        qubit_digits = "".join(str(q) for q in gate.qubits)
        key = (gate.name, qubit_digits)
        n = self._label_counts.get(key, 0)
        self._label_counts[key] = n + 1
        gate.label = f"Gate{gate.name}{qubit_digits}#{n}"

    def add(self, gate):
        qubits = gate.qubits

        # assign label once on first insertion
        self._assign_label(gate)

        if gate.t is not None:
            t = gate.t
            for q in qubits:
                self._ensure(q, t)
                if self.grid[q][t] is not None:
                    raise ValueError(f"Qubit {q} busy at t={t}")
        else:
            last = max(len(self.grid[q]) - 1 for q in qubits)
            top, bot = min(qubits), max(qubits)
            for q in range(top, bot + 1):
                last = max(last, len(self.grid[q]) - 1)

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

    def execute(self):
        depth = max(len(row) for row in self.grid)
        for t in range(depth):
            seen = set()
            gates_t = []
            for q in range(self.num_qubits):
                if t >= len(self.grid[q]):
                    continue
                g = self.grid[q][t]
                if g is not None and id(g) not in seen:
                    seen.add(id(g))
                    gates_t.append(g)
            print(f"t={t}: {[g.label for g in gates_t]}")


class GateOp:
    """Class representing quantum gate operations."""
    def __init__(self, name, qubits, t=None):
        self.name = name
        self.qubits = qubits
        self.t = t
        self.label = None  # filled by Circuit.add

    def __repr__(self):
        return self.label or self.name


circuit = Ciruit(3)
circuit.add(GateOp("H",    [0]))
circuit.add(GateOp("CNOT", [0, 1]))
circuit.add(GateOp("X",    [2]))
circuit.add(GateOp("Z",    [1]))
circuit.add(GateOp("CNOT", [0, 2]))

print(circuit.grid)
circuit.execute()
