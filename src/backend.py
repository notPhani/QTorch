import torch
import math
DTYPE = torch.complex64

class StateVectorBackend:
    def __init__(self, num_qubits, device = "cpu"):
        self.num_qubits = num_qubits
        self.dim = 1<<num_qubits
        self.device = device
        self.state = torch.zeros((self.dim,), dtype=DTYPE, device=device)

