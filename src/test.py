import time
import torch
from circuit import Circuit
from gates import GateOp
from backend import StatevectorBackend

def benchmark_circuit(circuit, num_qubits, persistent_data, warmup=2, runs=10):
    """Run a circuit multiple times and measure average performance."""
    
    # Warmup runs
    for _ in range(warmup):
        backend = StatevectorBackend(
            num_qubits, 
            device='cuda', 
            persistent_data=persistent_data,
            verbose=False  # Suppress prints in warmup
        )
        circuit.execute(backend)
    
    # Timed runs
    times = []
    for _ in range(runs):
        backend = StatevectorBackend(
            num_qubits, 
            device='cuda', 
            persistent_data=persistent_data,
            verbose=False  # Suppress prints in benchmark
        )
        
        start = time.perf_counter()
        final_state = circuit.execute(backend)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time)**2 for t in times) / len(times)) ** 0.5
    
    # Create one final backend to check cache stats
    final_backend = StatevectorBackend(
        num_qubits, 
        device='cuda', 
        persistent_data=persistent_data,
        verbose=True
    )
    circuit.execute(final_backend)
    
    return avg_time, std_time, final_backend

# Build test circuit
def build_test_circuit(num_qubits):
    circuit = Circuit(num_qubits)
    
    # Clifford encoding layer
    for i in range(num_qubits):
        circuit.add(GateOp("H", [i]))
    
    for i in range(num_qubits - 1):
        circuit.add(GateOp("CNOT", [i, i+1]))
    
    # Parametric ansatz layer
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
baseline_time, baseline_std, baseline_backend = benchmark_circuit(
    circuit, num_qubits, persistent_data=False, runs=10
)
print(f"Average time: {baseline_time*1000:.2f} ± {baseline_std*1000:.2f} ms")

print("\n" + "=" * 60)
print("OPTIMIZED (persistent_data=True)")
print("=" * 60)
optimized_time, optimized_std, optimized_backend = benchmark_circuit(
    circuit, num_qubits, persistent_data=True, runs=10
)
print(f"Average time: {optimized_time*1000:.2f} ± {optimized_std*1000:.2f} ms")

# Show cache stats
optimized_backend.print_cache_info()

# Comparison
print("=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"Speedup: {baseline_time / optimized_time:.2f}×")
print(f"Time saved: {(baseline_time - optimized_time)*1000:.2f} ms per run")
