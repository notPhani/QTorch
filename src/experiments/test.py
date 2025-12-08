import torch
import time
from typing import Sequence, Optional, List
from functools import lru_cache

DTYPE = torch.complex64


class GateSpec:
    def __init__(self, name, num_qubits, op_code, param_count, matrix_fn, doc=""):
        self.name = name
        self.num_qubits = num_qubits
        self.op_code = op_code
        self.param_count = param_count
        self.matrix_fn = matrix_fn
        self.__doc__ = doc


class GateOp:
    def __init__(self, name, qubits, params=None, depends_on=None, t=None):
        self.name = name
        self.qubits = list(qubits)
        # Store params on CPU (lightweight), move to device only when needed
        if params is not None:
            if isinstance(params, torch.Tensor):
                self.params = params.detach().cpu().flatten()
            else:
                self.params = torch.tensor(params, dtype=torch.float32).flatten()
        else:
            self.params = None
        self.depends_on = depends_on or []
        self.t = t
        self.spec: Optional[GateSpec] = None
        self.label: str = ""


# --- matrix builders (device-aware) ---

def _mat_I(_, device='cpu'):
    return torch.eye(2, dtype=DTYPE, device=device)


def _mat_X(_, device='cpu'):
    m = torch.zeros((2, 2), dtype=DTYPE, device=device)
    m[0, 1] = 1
    m[1, 0] = 1
    return m


def _mat_Y(_, device='cpu'):
    m = torch.zeros((2, 2), dtype=DTYPE, device=device)
    m[0, 1] = -1j
    m[1, 0] = 1j
    return m


def _mat_Z(_, device='cpu'):
    return torch.diag(torch.tensor([1, -1], dtype=DTYPE, device=device))


def _mat_H(_, device='cpu'):
    m = torch.tensor([[1, 1], [1, -1]], dtype=DTYPE, device=device)
    return m / torch.sqrt(torch.tensor(2.0, dtype=torch.float32, device=device))


def _mat_S(_, device='cpu'):
    return torch.diag(torch.tensor([1, 1j], dtype=DTYPE, device=device))


def _mat_Sdg(_, device='cpu'):
    return torch.diag(torch.tensor([1, -1j], dtype=DTYPE, device=device))


def _mat_T(_, device='cpu'):
    phase = torch.exp(1j * torch.tensor(torch.pi / 4, dtype=torch.float32, device=device))
    return torch.diag(torch.tensor([1, phase], dtype=DTYPE, device=device))


def _mat_Tdg(_, device='cpu'):
    phase = torch.exp(-1j * torch.tensor(torch.pi / 4, dtype=torch.float32, device=device))
    return torch.diag(torch.tensor([1, phase], dtype=DTYPE, device=device))


def _mat_Rx(params, device='cpu'):
    # params is already on device
    theta = params[0]
    c = torch.cos(theta / 2)
    s = -1j * torch.sin(theta / 2)
    return torch.stack([torch.stack([c, s]), torch.stack([s, c])], dim=0).to(DTYPE)


def _mat_Ry(params, device='cpu'):
    theta = params[0]
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])], dim=0).to(DTYPE)


def _mat_Rz(params, device='cpu'):
    theta = params[0]
    e_minus = torch.exp(-1j * theta / 2)
    e_plus = torch.exp(1j * theta / 2)
    return torch.diag(torch.stack([e_minus, e_plus]).to(DTYPE))


def _mat_Phase(params, device='cpu'):
    phi = params[0]
    # Create directly on device
    return torch.diag(torch.stack([torch.ones(1, dtype=DTYPE, device=params.device)[0], 
                                     torch.exp(1j * phi)]))


def _mat_CNOT(_, device='cpu'):
    m = torch.zeros((4, 4), dtype=DTYPE, device=device)
    m[0, 0] = 1
    m[1, 1] = 1
    m[2, 3] = 1
    m[3, 2] = 1
    return m


def _mat_CZ(_, device='cpu'):
    return torch.diag(torch.tensor([1, 1, 1, -1], dtype=DTYPE, device=device))


def _mat_SWAP(_, device='cpu'):
    m = torch.zeros((4, 4), dtype=DTYPE, device=device)
    m[0, 0] = 1
    m[1, 2] = 1
    m[2, 1] = 1
    m[3, 3] = 1
    return m


def _mat_iSWAP(_, device='cpu'):
    m = torch.zeros((4, 4), dtype=DTYPE, device=device)
    m[0, 0] = 1
    m[3, 3] = 1
    m[1, 2] = 1j
    m[2, 1] = 1j
    return m


def _mat_CP(params, device='cpu'):
    phi = params[0]
    vals = torch.ones(4, dtype=DTYPE, device=params.device)
    vals[3] = torch.exp(1j * phi)
    return torch.diag(vals)


def _mat_RXX(params, device='cpu'):
    theta = params[0]
    c = torch.cos(theta / 2)
    s = -1j * torch.sin(theta / 2)
    m = torch.zeros((4, 4), dtype=DTYPE, device=params.device)
    m[0, 0] = c
    m[0, 3] = s
    m[1, 1] = c
    m[1, 2] = s
    m[2, 1] = s
    m[2, 2] = c
    m[3, 0] = s
    m[3, 3] = c
    return m


def _mat_RYY(params, device='cpu'):
    theta = params[0]
    c = torch.cos(theta / 2)
    s = -1j * torch.sin(theta / 2)
    m = torch.zeros((4, 4), dtype=DTYPE, device=params.device)
    m[0, 0] = c
    m[0, 3] = -s
    m[1, 1] = c
    m[1, 2] = s
    m[2, 1] = s
    m[2, 2] = c
    m[3, 0] = -s
    m[3, 3] = c
    return m


def _mat_RZZ(params, device='cpu'):
    theta = params[0]
    e00 = torch.exp(-1j * theta / 2)
    e11 = torch.exp(1j * theta / 2)
    return torch.diag(torch.stack([e00, e11, e11, e00]).to(DTYPE))


# -------- Gate specs --------

_I_SPEC = GateSpec("I", 1, 0, 0, _mat_I, "Identity gate; leaves the qubit state unchanged.")
_X_SPEC = GateSpec("X", 1, 1, 0, _mat_X, "Pauli-X gate; bit flip: |0> <-> |1>.")
_Y_SPEC = GateSpec("Y", 1, 2, 0, _mat_Y, "Pauli-Y gate; bit and phase flip.")
_Z_SPEC = GateSpec("Z", 1, 3, 0, _mat_Z, "Pauli-Z gate; phase flip: |1> -> -|1>.")
_H_SPEC = GateSpec("H", 1, 4, 0, _mat_H, "Hadamard gate; creates superpositions.")
_S_SPEC = GateSpec("S", 1, 5, 0, _mat_S, "Phase gate; sqrt(Z).")
_SDG_SPEC = GateSpec("Sdg", 1, 6, 0, _mat_Sdg, "Inverse phase gate.")
_T_SPEC = GateSpec("T", 1, 7, 0, _mat_T, "T gate; non-Clifford.")
_TDG_SPEC = GateSpec("Tdg", 1, 8, 0, _mat_Tdg, "Inverse T gate.")

_RX_SPEC = GateSpec("Rx", 1, 10, 1, _mat_Rx, "Rotation around X.")
_RY_SPEC = GateSpec("Ry", 1, 11, 1, _mat_Ry, "Rotation around Y.")
_RZ_SPEC = GateSpec("Rz", 1, 12, 1, _mat_Rz, "Rotation around Z.")
_P_SPEC = GateSpec("P", 1, 13, 1, _mat_Phase, "Phase shift.")

_CNOT_SPEC = GateSpec("CNOT", 2, 20, 0, _mat_CNOT, "Controlled-NOT.")
_CZ_SPEC = GateSpec("CZ", 2, 21, 0, _mat_CZ, "Controlled-Z.")
_SWAP_SPEC = GateSpec("SWAP", 2, 22, 0, _mat_SWAP, "Swap.")
_ISWAP_SPEC = GateSpec("iSWAP", 2, 23, 0, _mat_iSWAP, "iSWAP.")

_CP_SPEC = GateSpec("CP", 2, 24, 1, _mat_CP, "Controlled phase.")
_RXX_SPEC = GateSpec("RXX", 2, 25, 1, _mat_RXX, "RXX.")
_RYY_SPEC = GateSpec("RYY", 2, 26, 1, _mat_RYY, "RYY.")
_RZZ_SPEC = GateSpec("RZZ", 2, 27, 1, _mat_RZZ, "RZZ.")


class Gates:
    class I: spec = _I_SPEC
    class X: spec = _X_SPEC
    class Y: spec = _Y_SPEC
    class Z: spec = _Z_SPEC
    class H: spec = _H_SPEC
    class S: spec = _S_SPEC
    class Sdg: spec = _SDG_SPEC
    class T: spec = _T_SPEC
    class Tdg: spec = _TDG_SPEC
    class Rx: spec = _RX_SPEC
    class Ry: spec = _RY_SPEC
    class Rz: spec = _RZ_SPEC
    class P: spec = _P_SPEC
    class CNOT: spec = _CNOT_SPEC
    class CZ: spec = _CZ_SPEC
    class SWAP: spec = _SWAP_SPEC
    class iSWAP: spec = _ISWAP_SPEC
    class CP: spec = _CP_SPEC
    class RXX: spec = _RXX_SPEC
    class RYY: spec = _RYY_SPEC
    class RZZ: spec = _RZZ_SPEC

    by_name = {
        "I": _I_SPEC, "X": _X_SPEC, "Y": _Y_SPEC, "Z": _Z_SPEC,
        "H": _H_SPEC, "S": _S_SPEC, "Sdg": _SDG_SPEC,
        "T": _T_SPEC, "Tdg": _TDG_SPEC,
        "Rx": _RX_SPEC, "Ry": _RY_SPEC, "Rz": _RZ_SPEC, "P": _P_SPEC,
        "CNOT": _CNOT_SPEC, "CZ": _CZ_SPEC, "SWAP": _SWAP_SPEC,
        "iSWAP": _ISWAP_SPEC,
        "CP": _CP_SPEC, "RXX": _RXX_SPEC, "RYY": _RYY_SPEC, "RZZ": _RZZ_SPEC,
    }

    by_opcode = {g.op_code: g for g in by_name.values()}


def _resolve_gate_spec(name: str):
    return Gates.by_name.get(name, None)


_original_init = GateOp.__init__


def _patched_init(self, name, qubits, params=None, depends_on=None, t=None):
    _original_init(self, name, qubits, params, depends_on, t)
    self.spec = _resolve_gate_spec(name)


GateOp.__init__ = _patched_init


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
        """Execute this circuit on the given backend."""
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


class StatevectorBackend:
    def __init__(
        self,
        num_qubits: int,
        device: Optional[str] = None,
        persistent_data: bool = True,
        angle_precision: float = 0.001,
        parametric_cache_size: int = 1024,
        verbose: bool = True,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.num_qubits = num_qubits
        self.device = torch.device(device)
        self.persistent_data = persistent_data
        self.angle_precision_val = angle_precision
        # Precision tensor on device for GPU-side quantization
        self.angle_precision = torch.tensor(angle_precision, dtype=torch.float32, device=self.device)
        self.verbose = verbose

        dim = 1 << num_qubits
        self.state = torch.zeros(dim, dtype=DTYPE, device=self.device)
        self.state[0] = 1
        self.creg = {}

        self._fixed_cache = {}
        self._parametric_cache = None

        if self.persistent_data:
            self._precompute_fixed_gates()
            self._setup_parametric_cache(parametric_cache_size)
        else:
            if self.verbose:
                print("[Backend] Caching DISABLED - running baseline mode")

    def _precompute_fixed_gates(self):
        """Precompute fixed gates directly on device - NO CPU involvement"""
        fixed_gate_names = [
            "H", "X", "Y", "Z", "S", "T",
            "Sdg", "Tdg",
            "CNOT", "CZ", "SWAP", "iSWAP",
            "I",
        ]
        cached_count = 0
        for gate_name in fixed_gate_names:
            if gate_name not in Gates.by_name:
                continue
            spec = Gates.by_name[gate_name]
            if spec.param_count != 0:
                continue
            try:
                # Create matrix directly on device
                matrix = spec.matrix_fn(None, device=self.device)
                self._fixed_cache[gate_name] = matrix
                cached_count += 1
            except Exception as e:
                if self.verbose:
                    print(f"[Warning] Failed to cache {gate_name}: {e}")
        if cached_count > 0 and self.verbose:
            print(f"[Backend] Precomputed {cached_count} fixed gates on {self.device}")

    def _setup_parametric_cache(self, maxsize: int):
        @lru_cache(maxsize=maxsize)
        def _cached_parametric_matrix(gate_name: str, quantized_params: tuple):
            """
            Only called on cache MISS.
            Converts tuple → tensor on device, computes matrix on device.
            """
            spec = Gates.by_name[gate_name]
            if quantized_params:
                # Rebuild tensor directly on device (no CPU intermediate)
                params_tensor = torch.tensor(quantized_params, dtype=torch.float32, device=self.device)
            else:
                params_tensor = None
            # Matrix created on device (params_tensor is on device)
            matrix = spec.matrix_fn(params_tensor, device=self.device)
            return matrix

        self._parametric_cache = _cached_parametric_matrix
        if self.verbose:
            print(f"[Backend] LRU cache enabled (size={maxsize}, precision={self.angle_precision_val} rad)")

    def _quantize_params(self, params: torch.Tensor) -> tuple:
        """
        Move to device, quantize on GPU, return hashable tuple.
        ONLY CPU transfer: final .cpu() for cache key (unavoidable).
        """
        if params is None or params.numel() == 0:
            return ()

        # Move to device if not already there (one-time cost per gate)
        if params.device != self.device:
            params = params.to(self.device)

        # Quantize entirely on GPU
        quantized = torch.round(params / self.angle_precision) * self.angle_precision

        # ONLY CPU transfer: for hashable cache key
        return tuple(float(x) for x in quantized.cpu())

    def apply_gate(self, gate_op: GateOp):
        """Apply gate - everything stays on device except cache key generation"""
        spec = gate_op.spec
        name = gate_op.name

        # Special ops
        if spec is None:
            if name == "M":
                self._apply_measure(gate_op.qubits[0])
            elif name == "Xc":
                self._apply_classical_pauli(gate_op.qubits[0], "X", gate_op.depends_on)
            elif name == "Zc":
                self._apply_classical_pauli(gate_op.qubits[0], "Z", gate_op.depends_on)
            return

        if self.persistent_data:
            # Priority 1: fixed cache (already on device)
            if name in self._fixed_cache:
                U = self._fixed_cache[name]

            # Priority 2: parametric cache
            elif gate_op.params is not None:
                # Quantize (moves to device internally, returns tuple for key)
                quantized_params = self._quantize_params(gate_op.params)
                # Cache returns matrix already on device
                U = self._parametric_cache(name, quantized_params)

            # Priority 3: non-parametric not cached
            else:
                U = spec.matrix_fn(None, device=self.device)
        else:
            # Baseline: compute fresh on device
            if gate_op.params is not None:
                params_on_device = gate_op.params.to(self.device)
                U = spec.matrix_fn(params_on_device, device=self.device)
            else:
                U = spec.matrix_fn(None, device=self.device)

        self._apply_k_qubit(U, gate_op.qubits)

    def _apply_k_qubit(self, U: torch.Tensor, targets: Sequence[int]):
        """All tensor ops on device"""
        n = self.num_qubits
        k = len(targets)
        assert U.shape == (1 << k, 1 << k)
        psi = self.state.view([2] * n)

        targets = list(targets)
        perm = [i for i in range(n) if i not in targets] + targets
        psi = psi.permute(perm)
        batch = psi.numel() // (1 << k)
        psi = psi.reshape(batch, 1 << k)
        psi = psi @ U.t()
        psi = psi.view([2] * n)

        inv = [0] * n
        for i, p in enumerate(perm):
            inv[p] = i
        psi = psi.permute(inv)
        self.state = psi.reshape(-1)

    def _apply_measure(self, q: int):
        """Measurement on device, one CPU transfer for probability"""
        n = self.num_qubits
        psi = self.state.view([2] * n)
        perm = [i for i in range(n) if i != q] + [q]
        psi = psi.permute(perm)
        psi = psi.reshape(-1, 2)
        probs = (psi.conj() * psi).sum(dim=0).real
        p0 = float(probs[0])  # Only CPU transfer
        r = torch.rand((), device=self.device)
        outcome = 0 if r < p0 else 1
        mask = torch.zeros_like(psi)
        mask[:, outcome] = 1
        psi = psi * mask
        norm = torch.linalg.norm(psi)
        if norm > 0:
            psi = psi / norm
        psi = psi.view([2] * n)
        inv = [0] * n
        for i, p in enumerate(perm):
            inv[p] = i
        psi = psi.permute(inv)
        self.state = psi.reshape(-1)
        self.creg[q] = outcome

    def _apply_classical_pauli(self, target_q: int, which: str, depends_on):
        """Classical control on device"""
        if len(depends_on) != 2:
            return
        m0_gate, m1_gate = depends_on
        q0 = m0_gate.qubits[0]
        q1 = m1_gate.qubits[0]
        b0 = self.creg.get(q0, None)
        b1 = self.creg.get(q1, None)
        if b0 is None or b1 is None:
            return
        fire = False
        if which == "Z":
            fire = (b0 == 1)
        elif which == "X":
            fire = (b1 == 1)
        else:
            raise ValueError(f"Unknown classical Pauli {which}")
        if not fire:
            return
        spec = Gates.by_name[which]
        U = spec.matrix_fn(None, device=self.device)
        self._apply_k_qubit(U, [target_q])

    def set_qubit_state(self, q: int, alpha: complex, beta: complex):
        """Initialize on device"""
        n = self.num_qubits
        vec = torch.zeros(1 << n, dtype=DTYPE, device=self.device)
        for bit in (0, 1):
            amp = alpha if bit == 0 else beta
            if amp == 0:
                continue
            idx = 0
            for j in range(n):
                idx <<= 1
                idx |= 1 if (j == q and bit == 1) else 0
            vec[idx] = amp
        norm = torch.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self.state = vec

    def set_state_vector(self, state: torch.Tensor):
        """Set state on device"""
        state = state.to(self.device, dtype=DTYPE).reshape(-1)
        dim = 1 << self.num_qubits
        if state.shape != (dim,):
            raise ValueError(f"State vector must have shape ({dim},), got {tuple(state.shape)}")
        norm = torch.linalg.norm(state)
        if norm == 0:
            raise ValueError("Cannot set state to zero vector")
        self.state = state / norm

    def get_cache_info(self):
        if not self.persistent_data:
            return {"enabled": False, "mode": "baseline"}
        info = {
            "enabled": True,
            "mode": "optimized",
            "fixed_gates_cached": len(self._fixed_cache),
            "fixed_gate_names": sorted(self._fixed_cache.keys()),
            "angle_precision": self.angle_precision_val,
            "device": str(self.device),
        }
        if self._parametric_cache is not None:
            cache_info = self._parametric_cache.cache_info()
            total = cache_info.hits + cache_info.misses
            hit_rate = cache_info.hits / total if total > 0 else 0
            info.update({
                "parametric_hits": cache_info.hits,
                "parametric_misses": cache_info.misses,
                "parametric_size": cache_info.currsize,
                "parametric_maxsize": cache_info.maxsize,
                "hit_rate": hit_rate,
                "hit_rate_percent": f"{hit_rate * 100:.2f}%",
            })
        return info

    def print_cache_info(self):
        info = self.get_cache_info()
        print("\n=== Backend Configuration ===")
        print(f"Persistent data: {info['enabled']}")
        print(f"Mode: {info['mode']}")
        if info["enabled"]:
            print("\n--- Fixed Gate Cache ---")
            print(f"Gates cached: {info['fixed_gates_cached']}")
            print(f"Device: {info['device']}")
            print(f"Gate list: {', '.join(info['fixed_gate_names'])}")
            if "parametric_hits" in info:
                print("\n--- Parametric LRU Cache ---")
                print(f"Hits: {info['parametric_hits']}")
                print(f"Misses: {info['parametric_misses']}")
                print(f"Hit rate: {info['hit_rate_percent']}")
                print(f"Cache size: {info['parametric_size']}/{info['parametric_maxsize']}")
                print(f"Angle precision: {info['angle_precision']} rad (~{info['angle_precision'] * 57.2958:.4f}°)")
        else:
            print("All gates computed fresh (baseline benchmarking)")
        print()

    def clear_parametric_cache(self):
        if self.persistent_data and self._parametric_cache is not None:
            self._parametric_cache.cache_clear()
            if self.verbose:
                print("[Backend] Parametric cache cleared")

    def reset_cache_stats(self):
        if self.persistent_data and self._parametric_cache is not None:
            if self.verbose:
                print("[Backend] Note: functools.lru_cache doesn't support stat reset without clearing")


class PhiManifoldExtractor:
    def __init__(self, circuit: Circuit, DecoherenceProjectionMatrix, BaseLinePauliOffset):
        self.circuit = circuit
        self.num_qubits = circuit.num_qubits
        self.grid = circuit.grid
        self.DecoherenceProjectionMatrix = DecoherenceProjectionMatrix
        self.BaseLinePauliOffset = BaseLinePauliOffset

    def compute_phi_manifold(self, t:int):
        # Phi manifold is a 6D feature vector per time step containing:
        # Phi[0] : memory term (alpha - lambda)*phi(t-1)
        # Phi[1] : Decoherence approximation term beta*(L<phi(t-1)>)
        # Phi[2] : Disturbance approximation term kappa*(L<D(t-1)>); where D(t-1) being aG(t-1) + b(M(t-1))
        # Phi[3] : is the Uniform exponential fall off in the circuit epsilon*Sum over all neighbours of phi(t-1) times exp(-distance norm)
        # Phi[4] : is the Non-linear interactions term rho*(phi(t-1))/(1 + phi(t-1)^2)
        # Phi[5] : is the Stochastic noise term eta(t-1)*sigma(G(f) + M(f))
        # where G(t-1), M(t-1) are specific scalars representing gate and measurement disturbances at time t-1
        # where G(f) and M(f) are just flags indicating whether a gate or measurement was applied at time t-1
        pass
        


    
