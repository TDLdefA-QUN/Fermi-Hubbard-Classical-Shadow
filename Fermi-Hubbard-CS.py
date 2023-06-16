import numpy as np
from qiskit_nature.second_q.hamiltonians.lattices import (
    BoundaryCondition,
    HyperCubicLattice,
    LineLattice,
    SquareLattice,
)
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.settings import QiskitNatureSettings
from qiskit_nature.second_q.problems import LatticeModelProblem
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.opflow import PauliSumOp
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms.optimizers import SPSA
from qiskit.utils import algorithm_globals
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit.algorithms.minimum_eigensolvers import VQE
import matplotlib.pyplot as plt
from qiskit.quantum_info.operators.symplectic.sparse_pauli_op import SparsePauliOp

# In order to use SparsePauliOp instead of PauliSumOp, we need to set the following
QiskitNatureSettings.use_pauli_sum_op = False

def create_fermi_hubbard_model():
    uniform_parameters = {"uniform_interaction": -1.0, "uniform_onsite_potential": 0.0}

    square_lattice = SquareLattice(
        rows=3, cols=2, boundary_condition=BoundaryCondition.PERIODIC
    )
    square_lattice = square_lattice.uniform_parameters(**uniform_parameters)

    size = (2, 2, 2)
    cubic_lattice = HyperCubicLattice(
        size=size, boundary_condition=BoundaryCondition.PERIODIC
    )
    cubic_lattice = cubic_lattice.uniform_parameters(**uniform_parameters)

    u = 5.0  # the interaction parameter U
    fhm = FermiHubbardModel(
        cubic_lattice,
        onsite_interaction=u,
    )

    return fhm


def find_ground_state_energy_numpy(hamiltonian_jw: SparsePauliOp) -> float:
    result = NumPyMinimumEigensolver().compute_minimum_eigenvalue(
        operator=PauliSumOp(hamiltonian_jw)
    )
    result = result.eigenvalue.real

    return result

def find_ground_state_energy_vqe(hamiltonian_jw: SparsePauliOp) -> float:
    iterations = 125
    ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")
    spsa = SPSA(maxiter=iterations)
    seed = 170
    algorithm_globals.random_seed = seed
    counts = []
    values = []

    def store_intermediate_result(eval_count, parameters, mean, std):
        counts.append(eval_count)
        values.append(mean)

    noiseless_estimator = AerEstimator(
        run_options={"seed": seed, "shots": 1024},
        transpile_options={"seed_transpiler": seed},
    )

    vqe = VQE(
        noiseless_estimator, ansatz, optimizer=spsa, callback=store_intermediate_result
    )
    result = vqe.compute_minimum_eigenvalue(operator=hamiltonian_jw)

    return result, counts, values


def plot_convergence(counts, values):
    plt.plot(counts, values)
    plt.xlabel("Eval count")
    plt.ylabel("Energy")
    plt.title("Convergence with no noise")
    plt.show()


def main():
    fhm = create_fermi_hubbard_model()

    ham = fhm.second_q_op().simplify()
    print("\n\nSecond quantized hamiltonian: \n", ham)

    hamiltonian_jw = JordanWignerMapper().map(ham)
    print("\n\nQubit hamiltonian: \n", hamiltonian_jw)
    print(f"\nNumber of qubits: {hamiltonian_jw.num_qubits}")

    ref_value = find_ground_state_energy_numpy(hamiltonian_jw)
    print(f"\nReference value: {ref_value:.5f}\n")

    result, counts, values = find_ground_state_energy_vqe(hamiltonian_jw)

    print(f"VQE on Aer qasm simulator (no noise): {result.eigenvalue.real:.5f}")
    print(
        f"Delta from reference energy value is {(result.eigenvalue.real - ref_value):.5f}"
    )

    plot_convergence(counts, values)


if __name__ == "__main__":
    main()
