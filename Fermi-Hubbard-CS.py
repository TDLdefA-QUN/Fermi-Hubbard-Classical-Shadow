from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import (
    BoundaryCondition,
    HyperCubicLattice,
    SquareLattice,
)
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.problems import LatticeModelProblem
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.opflow import PauliSumOp
from qiskit_nature.settings import QiskitNatureSettings
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms.optimizers import SPSA
from qiskit.utils import algorithm_globals
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit.algorithms.minimum_eigensolvers import VQE


t = -1.0  # the interaction parameter
v = 0.0  # the onsite potential
u = 5.0

square_lattice = SquareLattice(rows=3, cols=2, boundary_condition=BoundaryCondition.PERIODIC)

fhm_square = FermiHubbardModel(
    square_lattice.uniform_parameters(
        uniform_interaction=t,
        uniform_onsite_potential=v,
    ),
    onsite_interaction=u,
)




size = (2, 2, 2)
boundary_condition = (
    BoundaryCondition.PERIODIC,
    BoundaryCondition.PERIODIC,
    BoundaryCondition.PERIODIC,
)
cubic_lattice = HyperCubicLattice(size=size, boundary_condition=boundary_condition)


QiskitNatureSettings.use_pauli_sum_op = False

t = -1.0  # the interaction parameter
v = 0.0  # the onsite potential
u = 5.0  # the interaction parameter U

fhm_cubic = FermiHubbardModel(
    cubic_lattice.uniform_parameters(
        uniform_interaction=t,
        uniform_onsite_potential=v, 
    ),
    onsite_interaction=u,
)

ham = fhm_cubic.second_q_op()
print(ham)



lmp = LatticeModelProblem(fhm_cubic)

hamiltonian_jw = JordanWignerMapper().map(ham)
#print(np.count_nonzero(hamiltonian_jw - np.diag(np.diagonal(hamiltonian_jw)))
print(hamiltonian_jw)


numpy_solver = NumPyMinimumEigensolver()
result = numpy_solver.compute_minimum_eigenvalue(operator=PauliSumOp(hamiltonian_jw))
ref_value = result.eigenvalue.real
print(f"Reference value: {ref_value:.5f}")

iterations = 125
ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")
spsa = SPSA(maxiter=iterations)
counts = []
values = []


def store_intermediate_result(eval_count, parameters, mean, std):
    counts.append(eval_count)
    values.append(mean)


seed = 170
algorithm_globals.random_seed = seed

noiseless_estimator = AerEstimator(
    run_options={"seed": seed, "shots": 1024},
    transpile_options={"seed_transpiler": seed},
    backend_options={"method":"density_matrix"}

)




vqe = VQE(
    noiseless_estimator, ansatz, optimizer=spsa, callback=store_intermediate_result
)
result = vqe.compute_minimum_eigenvalue(operator=hamiltonian_jw)

print(f"VQE on Aer qasm simulator (no noise): {result.eigenvalue.real:.5f}")
print(
    f"Delta from reference energy value is {(result.eigenvalue.real - ref_value):.5f}"
)
import matplotlib.pyplot as plt


plt.plot(counts, values)
plt.xlabel("Eval count")
plt.ylabel("Energy")
plt.title("Convergence with no noise")

plt.show()
print(f'Number of qubits: {hamiltonian_jw.num_qubits}')
import psutil
print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
