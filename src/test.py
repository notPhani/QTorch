import time
import torch
from circuit import Circuit
from gates import GateOp
from backend import StatevectorBackend

def benchmark_circuit(circuit, num_qubits, persistent_data, warmup=2, runs=10):
    """Run a circuit multiple times and measure average performance."""
    
    # Create backend with specified mode
    backend = StatevectorBackend(num_qubits, device='cuda', persistent_data=persistent_data)
    
    # Warmup runs (for GPU kernel compilation, cache warming)
    for _ in range(warmup):
        backend_copy = StatevectorBackend(num_qubits, device='cuda', persistent_data=persistent_data)
        circuit.execute(backend_copy)
    
    # Timed runs
    times = []
    for _ in range(runs):
        backend_copy = StatevectorBackend(num_qubits, device='cuda', persistent_data=persistent_data)
        
        start = time.perf_counter()
        final_state = circuit.execute(backend_copy)
        torch.cuda.synchronize()  # Wait for GPU
        elapsed = time.perf_counter() - start
        
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time)**2 for t in times) / len(times)) ** 0.5
    
    return avg_time, std_time, final_state

# Build a test circuit
def build_test_circuit(num_qubits):
    circuit = Circuit(num_qubits)
    
    # Clifford encoding layer
    for i in range(num_qubits):
        circuit.add(GateOp("H", [i]))
    
    for i in range(num_qubits - 1):
        circuit.add(GateOp("CNOT", [i, i+1]))
    
    # Parametric ansatz layer
    import math
    for i in range(num_qubits):
        circuit.add(GateOp("Rx", [i], params=[0.5]))
        circuit.add(GateOp("Ry", [i], params=[1.2]))
    
    for i in range(num_qubits - 1):
        circuit.add(GateOp("CNOT", [i, i+1]))
    
    return circuit

# Run benchmarks
num_qubits = 20
circuit = build_test_circuit(num_qubits)

print("=" * 60)
print("BASELINE (persistent_data=False)")
print("=" * 60)
baseline_time, baseline_std, baseline_state = benchmark_circuit(
    circuit, num_qubits, persistent_data=False, runs=10
)
print(f"Average time: {baseline_time*1000:.2f} ± {baseline_std*1000:.2f} ms")
print(f"State norm: {torch.linalg.norm(baseline_state):.8f}")

print("\n" + "=" * 60)
print("OPTIMIZED (persistent_data=True)")
print("=" * 60)
optimized_time, optimized_std, optimized_state = benchmark_circuit(
    circuit, num_qubits, persistent_data=True, runs=10
)
print(f"Average time: {optimized_time*1000:.2f} ± {optimized_std*1000:.2f} ms")
print(f"State norm: {torch.linalg.norm(optimized_state):.8f}")

state_diff = torch.linalg.norm(baseline_state - optimized_state)
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"Speedup: {baseline_time / optimized_time:.2f}×")
print(f"State difference: {state_diff:.2e} (should be ~0)")
print(f"Correctness: {'✓ PASS' if state_diff < 1e-6 else '✗ FAIL'}")
