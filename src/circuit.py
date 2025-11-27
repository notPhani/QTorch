from gates import Gates
class Ciruit:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.grid = [[] for _ in range(num_qubits)]
        # (name, qubit_digits) -> count
        self.label_counts = {}

    def _ensure(self, q, t):
        while len(self.grid[q]) <= t:
            self.grid[q].append(None)

    def _assign_label(self, gate):
        # qubits in the exact order given → direction preserved
        qubit_digits = "".join(str(q) for q in gate.qubits)
        key = (gate.name, qubit_digits)
        n = self.label_counts.get(key, 0)
        self.label_counts[key] = n + 1
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
            for g in gates_t:
                print(f"Executing {g} on qubits {g.qubits}")
                

class GateOp:
    """Class representing quantum gate operations."""
    def __init__(self, name, qubits, depends_on = None, t=None):
        self.name = name
        self.qubits = qubits
        self.t = t
        self.depends_on = depends_on or []
        self.label = None  # filled by Circuit.add
    def __repr__(self):
        return self.label or self.name


def draw_ascii(circ: Ciruit, show_t=True, min_width=5, pad=2):
    """
    ASCII renderer with dynamic cell width.

    min_width: minimum width of each cell.
    pad: extra padding characters beyond the longest label.
    """

    n = circ.num_qubits
    depth = max((len(row) for row in circ.grid), default=0)

    # Normalize all rows to same depth
    for q in range(n):
        while len(circ.grid[q]) < depth:
            circ.grid[q].append(None)

    # Collect all labels/names to decide width
    labels = []
    for t in range(depth):
        seen = set()
        for q in range(n):
            g = circ.grid[q][t]
            if g is not None and id(g) not in seen:
                seen.add(id(g))
                lab = getattr(g, "label", None) or g.name
                # Strip "Gate" prefix for display if you like
                if lab.startswith("Gate"):
                    lab = lab[4:]
                labels.append(lab)

    max_lab_len = max((len(s) for s in labels), default=0)
    inner_width = max(min_width - 2, max_lab_len + pad)
    width = inner_width + 2  # account for [ ]

    cell_empty = "─" * width

    def gate_sym(g: GateOp):
        base = getattr(g, "label", None) or g.name
        if base.startswith("Gate"):
            base = base[4:]
        return base[:inner_width].center(inner_width)

    # Initialize lines
    lines = [[cell_empty for _ in range(depth)] for _ in range(n)]

    # First pass: place gate boxes
    for t in range(depth):
        seen = set()
        gates_t = []
        for q in range(n):
            g = circ.grid[q][t]
            if g is not None and id(g) not in seen:
                seen.add(id(g))
                gates_t.append(g)

        for g in gates_t:
            qs = sorted(g.qubits)
            sym = gate_sym(g)
            if len(qs) == 1:
                q = qs[0]
                lines[q][t] = "[" + sym + "]"
            else:
                for q in qs:
                    lines[q][t] = "[" + sym + "]"

    # Second pass: draw vertical connectors
    for t in range(depth):
        seen = set()
        gates_t = []
        for q in range(n):
            g = circ.grid[q][t]
            if g is not None and id(g) not in seen:
                seen.add(id(g))
                gates_t.append(g)

        for g in gates_t:
            qs = sorted(g.qubits)
            if len(qs) <= 1:
                continue
            top, bot = qs[0], qs[-1]
            for q in range(top + 1, bot):
                if q not in qs and lines[q][t] == cell_empty:
                    lines[q][t] = "│".center(width)

    # Optional time axis
    if show_t:
        header = "    " + " ".join(f"{t:^{width}}" for t in range(depth))
        print(header)

    for q in range(n):
        row = " ".join(lines[q])
        print(f"q{q}: {row}")

c = Ciruit(3)

# 0: |ψ>, 1: Alice, 2: Bob

# 1) Bell pair between 1 and 2
c.add(GateOp("H",    [1]))        # H on 1
c.add(GateOp("CNOT", [1, 2]))     # CNOT 1->2

# 2) Entangle data qubit 0 with Alice's half
c.add(GateOp("CNOT", [0, 1]))     # CNOT 0->1
c.add(GateOp("H",    [0]))        # H on 0

# 3) Measure 0 and 1
m0 = GateOp("M", [0])
m1 = GateOp("M", [1])
c.add(m0)
c.add(m1)

# 4) Classically controlled corrections on Bob (2),
#    depending on both measurements
zc = GateOp("Zc", [2], depends_on=[m0, m1])
xc = GateOp("Xc", [2], depends_on=[m0, m1])
c.add(zc)
c.add(xc)

# --- visualize ---
draw_ascii(c)
print("\nExecute:")
c.execute()
