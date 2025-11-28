import torch
from typing import Sequence, Optional
from gates import GateOp, Gates

DTYPE = torch.complex64

class StatevectorBackend:
    def __init__(
        self, 
        num_qubits: int, 
        device: Optional[str] = None,
        persistent_data: bool = True  # NEW: Master cache toggle
    ):
        """
        Statevector backend with optional caching optimizations.
        
        Args:
            num_qubits: Number of qubits
            device: 'cuda', 'cpu', or None (auto-detect)
            persistent_data: Enable all caching optimizations (fixed gates, LRU, quantization)
                            Set to False for baseline benchmarking
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.num_qubits = num_qubits
        self.device = device
        self.persistent_data = persistent_data  # Store the flag
        
        # Initialize state vector
        dim = 1 << num_qubits
        self.state = torch.zeros(dim, dtype=DTYPE, device=device)
        self.state[0] = 1
        self.creg = {}  # qubit index -> classical bit
        
        # Setup caching if enabled
        self._fixed_cache = {}
        if self.persistent_data:
            self._precompute_fixed_gates()
        else:
            print(f"[Backend] Caching DISABLED - running baseline mode")
    
    def _precompute_fixed_gates(self):
        """
        Precompute and cache all non-parametric gate matrices.
        Only runs if persistent_data=True.
        """
        fixed_gate_names = [
            # Single-qubit Cliffords
            'H', 'X', 'Y', 'Z', 'S', 'T',
            
            # Inverses
            'Sdg', 'Tdg',
            
            # Two-qubit gates
            'CNOT', 'CZ', 'CY', 'SWAP', 'iSWAP',
            
            # Multi-qubit gates
            'Toffoli', 'Fredkin',
            
            # Identity
            'I',
        ]
        
        cached_count = 0
        
        for gate_name in fixed_gate_names:
            if gate_name not in Gates.by_name:
                continue
            
            spec = Gates.by_name[gate_name]
            
            if spec.param_count != 0:
                continue
            
            try:
                matrix = spec.matrix_fn(None)
                self._fixed_cache[gate_name] = matrix.to(self.device)
                cached_count += 1
                
            except Exception as e:
                print(f"[Warning] Failed to cache {gate_name}: {e}")
        
        print(f"[Backend] Precomputed {cached_count} fixed gates (persistent_data=True)")
    
    def apply_gate(self, gate_op: GateOp):
        """Apply a gate operation to the state vector."""
        spec = gate_op.spec
        name = gate_op.name
        
        # Handle special ops (measurements, classical control)
        if spec is None:
            if name == "M":
                self._apply_measure(gate_op.qubits[0])
            elif name == "Xc":
                self._apply_classical_pauli(gate_op.qubits[0], "X", gate_op.depends_on)
            elif name == "Zc":
                self._apply_classical_pauli(gate_op.qubits[0], "Z", gate_op.depends_on)
            return
        
        # Get gate matrix with or without caching
        if self.persistent_data and name in self._fixed_cache:
            # Fast path: use cached matrix
            U = self._fixed_cache[name]
        else:
            # Baseline path: compute fresh every time
            U = spec.matrix_fn(gate_op.params).to(self.device)
        
        self._apply_k_qubit(U, gate_op.qubits)
    
    def _apply_k_qubit(self, U: torch.Tensor, targets: Sequence[int]):
        """Apply k-qubit gate U (2^k x 2^k) on given target qubits."""
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
        """Z-basis measurement with collapse."""
        n = self.num_qubits
        psi = self.state.view([2] * n)
        
        perm = [i for i in range(n) if i != q] + [q]
        psi = psi.permute(perm)
        psi = psi.reshape(-1, 2)
        
        probs = (psi.conj() * psi).sum(dim=0).real
        p0 = float(probs[0])
        
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
        """Apply X or Z conditioned on measurement outcomes."""
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
        U = spec.matrix_fn(None).to(self.device)
        self._apply_k_qubit(U, [target_q])
    
    def set_qubit_state(self, q: int, alpha: complex, beta: complex):
        """
        Initialize qubit q to alpha|0> + beta|1>, with others in |0>.
        Only correct if the current state is |0...0>.
        """
        n = self.num_qubits
        vec = torch.zeros(1 << n, dtype=DTYPE, device=self.device)
        
        for bit in (0, 1):
            amp = alpha if bit == 0 else beta
            if amp == 0:
                continue
            
            idx = 0
            for j in range(n):
                idx <<= 1
                idx |= (1 if (j == q and bit == 1) else 0)
            vec[idx] = amp
        
        norm = torch.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self.state = vec
    
    def set_state_vector(self, state: torch.Tensor):
        """
        Set the state vector directly.
        state: complex tensor of shape (2**num_qubits,) or (2**num_qubits, 1).
        It will be normalized and moved to the backend's device / dtype.
        """
        state = state.to(self.device, dtype=DTYPE).reshape(-1)
        dim = 1 << self.num_qubits
        
        if state.shape != (dim,):
            raise ValueError(f"State vector must have shape ({dim},), got {tuple(state.shape)}")
        
        norm = torch.linalg.norm(state)
        if norm == 0:
            raise ValueError("Cannot set state to zero vector")
        
        self.state = state / norm
    
    # ===== Introspection =====
    
    def get_cache_info(self):
        """Get info about caching status."""
        if not self.persistent_data:
            return {
                'enabled': False,
                'mode': 'baseline'
            }
        
        return {
            'enabled': True,
            'mode': 'optimized',
            'fixed_gates_cached': len(self._fixed_cache),
            'gate_names': sorted(self._fixed_cache.keys()),
            'device': self.device
        }
    
    def print_cache_info(self):
        """Pretty-print cache info."""
        info = self.get_cache_info()
        
        print("\n=== Backend Configuration ===")
        print(f"Persistent data: {info['enabled']}")
        print(f"Mode: {info['mode']}")
        
        if info['enabled']:
            print(f"Fixed gates cached: {info['fixed_gates_cached']}")
            print(f"Device: {info['device']}")
            print(f"Cached gates: {', '.join(info['gate_names'])}")
        else:
            print("All gates computed fresh (baseline benchmarking)")
        print()
