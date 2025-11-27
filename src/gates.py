from dataclasses import dataclass
from typing import Callable, Sequence, Optional
import math
import torch

DTYPE = torch.complex64

@dataclass(frozen=True)
class GateSpec:
    name: str
    arity: int
    op_code: int
    param_count: int          # 0 = fixed, >0 = number of parameters
    matrix_fn: Callable[[Optional[Sequence[float]]], torch.Tensor]
    description: str
#-----------------------------------------------Single-qubit gates-----------------------------------------------
def _mat_I(_=None):
    return torch.eye(2, dtype=DTYPE)

def _mat_X(_=None):
    return torch.tensor([[0, 1],
                         [1, 0]], dtype=DTYPE)

def _mat_Y(_=None):
    return torch.tensor([[0, -1j],
                         [1j, 0]], dtype=DTYPE)

def _mat_Z(_=None):
    return torch.tensor([[1, 0],
                         [0,-1]], dtype=DTYPE)

def _mat_H(_=None):
    s = 1 / math.sqrt(2.0)
    return torch.tensor([[ s,  s],
                         [ s, -s]], dtype=DTYPE)

def _mat_S(_=None):
    return torch.tensor([[1, 0],
                         [0, 1j]], dtype=DTYPE)

def _mat_Sdg(_=None):
    return torch.tensor([[1, 0],
                         [0, -1j]], dtype=DTYPE)

def _mat_T(_=None):
    return torch.tensor([[1, 0],
                         [0, torch.exp(1j * math.pi/4)]], dtype=DTYPE)

def _mat_Tdg(_=None):
    return torch.tensor([[1, 0],
                         [0, torch.exp(-1j * math.pi/4)]], dtype=DTYPE)

#-----------------------------------------------Parameterized single-qubit gates-----------------------------------------------
def _mat_Rx(params):
    (theta,) = params
    c = math.cos(theta/2)
    s = math.sin(theta/2)
    return torch.tensor([[c, -1j*s],
                         [-1j*s, c]], dtype=DTYPE)

def _mat_Ry(params):
    (theta,) = params
    c = math.cos(theta/2)
    s = math.sin(theta/2)
    return torch.tensor([[c, -s],
                         [s,  c]], dtype=DTYPE)

def _mat_Rz(params):
    (theta,) = params
    return torch.tensor([[torch.exp(-0.5j*theta), 0],
                         [0, torch.exp(0.5j*theta)]], dtype=DTYPE)

def _mat_Phase(params):
    (phi,) = params
    return torch.tensor([[1, 0],
                         [0, torch.exp(1j*phi)]], dtype=DTYPE)

#-----------------------------------------------Two-qubit gates-----------------------------------------------
def _mat_CNOT(_=None):
    return torch.tensor([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,0,1],
        [0,0,1,0],
    ], dtype=DTYPE)

def _mat_CZ(_=None):
    return torch.tensor([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,-1],
    ], dtype=DTYPE)

def _mat_SWAP(_=None):
    return torch.tensor([
        [1,0,0,0],
        [0,0,1,0],
        [0,1,0,0],
        [0,0,0,1],
    ], dtype=DTYPE)

def _mat_iSWAP(_=None):
    return torch.tensor([
        [1,0,0,0],
        [0,0,1j,0],
        [0,1j,0,0],
        [0,0,0,1],
    ], dtype=DTYPE)
#-----------------------------------------------Parameterized two-qubit gates-----------------------------------------------
def _mat_CP(params):
    (phi,) = params
    return torch.tensor([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0, torch.exp(1j*phi)],
    ], dtype=DTYPE)

def _mat_RXX(params):
    (theta,) = params
    c = math.cos(theta/2)
    s = -1j * math.sin(theta/2)
    return torch.tensor([
        [c, 0, 0, s],
        [0, c, s, 0],
        [0, s, c, 0],
        [s, 0, 0, c],
    ], dtype=DTYPE)

def _mat_RYY(params):
    (theta,) = params
    c = math.cos(theta/2)
    s = -1j * math.sin(theta/2)
    return torch.tensor([
        [ c, 0, 0,-s],
        [ 0, c, s, 0],
        [ 0, s, c, 0],
        [-s, 0, 0, c],
    ], dtype=DTYPE)

def _mat_RZZ(params):
    (theta,) = params
    e = torch.exp(-1j * theta/2)
    ep = torch.exp(1j * theta/2)
    return torch.tensor([
        [e, 0, 0, 0],
        [0, ep, 0, 0],
        [0, 0, ep, 0],
        [0, 0, 0, e],
    ], dtype=DTYPE)


# 1-qubit fixed
_I_SPEC    = GateSpec("I",    1,  0, 0, _mat_I,
    "Identity gate; leaves the qubit state unchanged.")
_X_SPEC    = GateSpec("X",    1,  1, 0, _mat_X,
    "Pauli-X gate; bit flip: |0> <-> |1>.")
_Y_SPEC    = GateSpec("Y",    1,  2, 0, _mat_Y,
    "Pauli-Y gate; bit and phase flip.")
_Z_SPEC    = GateSpec("Z",    1,  3, 0, _mat_Z,
    "Pauli-Z gate; phase flip: |1> -> -|1>.")
_H_SPEC    = GateSpec("H",    1,  4, 0, _mat_H,
    "Hadamard gate; creates superpositions between |0> and |1>.")
_S_SPEC    = GateSpec("S",    1,  5, 0, _mat_S,
    "Phase gate; applies a +π/2 phase to |1> (sqrt(Z)).")
_SDG_SPEC  = GateSpec("Sdg",  1,  6, 0, _mat_Sdg,
    "Inverse phase gate; applies a -π/2 phase to |1>.")
_T_SPEC    = GateSpec("T",    1,  7, 0, _mat_T,
    "T gate; π/4 phase on |1>, a non-Clifford single-qubit gate.")
_TDG_SPEC  = GateSpec("Tdg",  1,  8, 0, _mat_Tdg,
    "Inverse T gate; -π/4 phase on |1>.")


# 1-qubit param
_RX_SPEC   = GateSpec("Rx",   1, 10, 1, _mat_Rx,
    "Rotation around the X axis by angle theta.")
_RY_SPEC   = GateSpec("Ry",   1, 11, 1, _mat_Ry,
    "Rotation around the Y axis by angle theta.")
_RZ_SPEC   = GateSpec("Rz",   1, 12, 1, _mat_Rz,
    "Rotation around the Z axis by angle theta.")
_P_SPEC    = GateSpec("P",    1, 13, 1, _mat_Phase,
    "Single-qubit phase shift by angle phi (equivalent to Rz up to global phase).")


# 2-qubit fixed
_CNOT_SPEC = GateSpec("CNOT", 2, 20, 0, _mat_CNOT,
    "Controlled-NOT gate; flips the target qubit when the control is |1>.")
_CZ_SPEC   = GateSpec("CZ",   2, 21, 0, _mat_CZ,
    "Controlled-Z gate; applies Z to the target when control is |1>.")
_SWAP_SPEC = GateSpec("SWAP", 2, 22, 0, _mat_SWAP,
    "Swap gate; exchanges the states of two qubits.")
_ISWAP_SPEC= GateSpec("iSWAP",2, 23, 0, _mat_iSWAP,
    "iSWAP gate; swaps |01> and |10> and adds a phase of i.")


# 2-qubit param
_CP_SPEC   = GateSpec("CP",   2, 24, 1, _mat_CP,
    "Controlled phase gate; adds a phase e^{i phi} to |11>.")
_RXX_SPEC  = GateSpec("RXX",  2, 25, 1, _mat_RXX,
    "Two-qubit XX interaction rotation by angle theta.")
_RYY_SPEC  = GateSpec("RYY",  2, 26, 1, _mat_RYY,
    "Two-qubit YY interaction rotation by angle theta.")
_RZZ_SPEC  = GateSpec("RZZ",  2, 27, 1, _mat_RZZ,
    "Two-qubit ZZ interaction rotation by angle theta.")


class Gates:
    class I:    spec = _I_SPEC



class Gates:
    class I:    spec = _I_SPEC
    class X:    spec = _X_SPEC
    class Y:    spec = _Y_SPEC
    class Z:    spec = _Z_SPEC
    class H:    spec = _H_SPEC
    class S:    spec = _S_SPEC
    class Sdg:  spec = _SDG_SPEC
    class T:    spec = _T_SPEC
    class Tdg:  spec = _TDG_SPEC

    class Rx:   spec = _RX_SPEC
    class Ry:   spec = _RY_SPEC
    class Rz:   spec = _RZ_SPEC
    class P:    spec = _P_SPEC

    class CNOT: spec = _CNOT_SPEC
    class CZ:   spec = _CZ_SPEC
    class SWAP: spec = _SWAP_SPEC
    class iSWAP:spec = _ISWAP_SPEC

    class CP:   spec = _CP_SPEC
    class RXX:  spec = _RXX_SPEC
    class RYY:  spec = _RYY_SPEC
    class RZZ:  spec = _RZZ_SPEC

    by_name = {
        "I": _I_SPEC,
        "X": _X_SPEC,
        "Y": _Y_SPEC,
        "Z": _Z_SPEC,
        "H": _H_SPEC,
        "S": _S_SPEC,
        "Sdg": _SDG_SPEC,
        "T": _T_SPEC,
        "Tdg": _TDG_SPEC,
        "Rx": _RX_SPEC,
        "Ry": _RY_SPEC,
        "Rz": _RZ_SPEC,
        "P": _P_SPEC,
        "CNOT": _CNOT_SPEC,
        "CZ": _CZ_SPEC,
        "SWAP": _SWAP_SPEC,
        "iSWAP": _ISWAP_SPEC,
        "CP": _CP_SPEC,
        "RXX": _RXX_SPEC,
        "RYY": _RYY_SPEC,
        "RZZ": _RZZ_SPEC,
    }

    by_opcode = {
        g.op_code: g for g in by_name.values()
    }
