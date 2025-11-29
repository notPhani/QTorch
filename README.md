**What is QtorchX?**

QtorchX is a noise-aware quantum simulation research framework designed for the NISQ era.
It focuses on accurately modeling real hardware noise dynamics while still enabling compiler-level acceleration.

Current simulators face a tradeoff:

| Simulator              | Fast? | Realistic Noise?  |
| ---------------------- | :---: | :--------------:  |
| Ideal state simulators |   ✔   |         ✗        |
| Density matrix sim     |   ✗   |         ✔        |
| Simple noise channels  |   ✔   |         ✗        |

QtorchX aims to deliver both: Physical noise fidelity + Gate fusion performance
It serves as a foundation for future quantum machine learning research that depends on realistic noise behavior.
