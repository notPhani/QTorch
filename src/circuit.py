from gates import GateOp
class Ciruit:
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
            qubits = gate.qubits
            last = max(len(self.grid[q]) - 1 for q in qubits)
            # also block on any gate in between, if you want visual cleanliness:
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


circuit = Ciruit(3)
circuit.add(GateOp("H", [0]))
circuit.add(GateOp("CNOT", [0, 1]))
circuit.add(GateOp("X", [2]))
circuit.add(GateOp("Z", [1]))
circuit.add(GateOp("CNOT", [0,2]))

print(circuit.grid)

