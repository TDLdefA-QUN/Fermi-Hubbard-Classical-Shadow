from qiskit.algorithms.minimum_eigensolvers import AdaptVQE, VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from qiskit.circuit.library import EvolvedOperatorAnsatz

import Fermi_Hubbard_CS as fhcs
from qiskit_nature.second_q.mappers import JordanWignerMapper
# from qiskit.circuit.library import TwoLocal

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp

from qiskit.circuit import QuantumCircuit, QuantumRegister

fhm = fhcs.linear_FHM(t=-1.0, v=0.0, u=5.0, size=2)
hamiltonian = JordanWignerMapper().map(fhm.second_q_op())

num_qubits = hamiltonian.num_qubits

# This code uses Qiskit 0.44

# REFERENCE: https://www.nature.com/articles/s41467-019-10988-2

# Above is the standard code for obtaining the hamiltonian
# Below is taken from the Qiskit's test code for adapt-vqe
# See: https://github.com/Qiskit/qiskit/blob/main/test/python/algorithms/minimum_eigensolvers/test_adapt_vqe.py#L64

# Operator or Excitation pool contains the single and double excitations
# The following pool most likely may not be suitable for Fermi-Hubbard Model since it is taken from the test code
op_pool = [
    PauliSumOp(
        SparsePauliOp(["IIIY", "IIZY"], coeffs=[0.5 + 0.0j, -0.5 + 0.0j]), coeff=1.0
    ),
    PauliSumOp(
        SparsePauliOp(["ZYII", "IYZI"], coeffs=[-0.5 + 0.0j, 0.5 + 0.0j]), coeff=1.0
    ),
    PauliSumOp(
        SparsePauliOp(
            ["ZXZY", "IXIY", "IYIX", "ZYZX", "IYZX", "ZYIX", "ZXIY", "IXZY"],
            coeffs=[
                -0.125 + 0.0j,
                0.125 + 0.0j,
                -0.125 + 0.0j,
                0.125 + 0.0j,
                0.125 + 0.0j,
                -0.125 + 0.0j,
                0.125 + 0.0j,
                -0.125 + 0.0j,
            ],
        ),
        coeff=1.0,
    ),
]

# Initial state is going to be a HF State
initial_state = QuantumCircuit(QuantumRegister(4))
initial_state.x(0)
initial_state.x(1)


ansatz = EvolvedOperatorAnsatz(op_pool,initial_state=initial_state)

vqe = VQE(Estimator(), ansatz, SLSQP())

adapt_vqe = AdaptVQE(vqe,threshold=1e-15) # 1e-15 is the minimum threshold 

result = adapt_vqe.compute_minimum_eigenvalue(hamiltonian)

print(result)