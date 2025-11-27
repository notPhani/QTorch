from dataclasses import dataclass
from typing import Callable, Sequence, Optional, List
from backend import StatevectorBackend
import math
import torch
from gates import *

class GateOp:
    """Quantum gate or special op in a circuit."""
    def __init__(self, name: str, qubits: List[int],
                 params: Optional[Sequence[float]] = None,
                 depends_on: Optional[List["GateOp"]] = None,
                 t: Optional[int] = None):
        self.name = name
        self.qubits = list(qubits)
        self.params = list(params) if params is not None else []
        self.depends_on = depends_on or []
        self.t = t
        self.label = None
        self.spec = Gates.by_name.get(name, None)

    def __repr__(self):
        return self.label or self.name

class Circuit:
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.grid = [[] for _ in range(num_qubits)]
        self.label_counts = {}

    def _ensure(self, q, t):
        while len(self.grid[q]) <= t:
            self.grid[q].append(None)

    def _assign_label(self, gate: GateOp):
        qubit_digits = "".join(str(q) for q in gate.qubits)
        key = (gate.name, qubit_digits)
        n = self.label_counts.get(key, 0)
        self.label_counts[key] = n + 1
        gate.label = f"Gate{gate.name}{qubit_digits}#{n}"

    def add(self, gate: GateOp):
        qubits = gate.qubits
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
            for parent in (gate.depends_on or []):
                if parent.t is not None:
                    last = max(last, parent.t)
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

    def execute(self, backend) -> torch.Tensor:
        if backend.num_qubits != self.num_qubits:
            raise ValueError("Backend qubit count mismatch")
        depth = max((len(row) for row in self.grid), default=0)
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
            for g in gates_t:
                backend.apply_gate(g)
        return backend.state
    
   