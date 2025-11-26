import torch

class GateOp:
    """Class representing quantum gate operations."""
    def __init__(self, name, qubits, t = None):
        self.name = name
        self.qubits = qubits
        self.t = t
    def __repr__(self):
        return self.name

class 