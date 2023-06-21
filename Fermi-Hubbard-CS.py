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
from qiskit.algorithms.optimizers import SLSQP, SPSA
from qiskit.utils import algorithm_globals
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import Optimizer, OptimizerResult, OptimizerState
from qiskit import Aer, transpile
import psutil
import pennylane as qml
import pennylane.numpy as np
import matplotlib.pyplot as plt
import time
import argparse


def cubic_FHM(t, v, u, size):
    '''
    Second Quantized Hamiltonian for the Fermi-Hubbard model in a periodic cube
    Returns the hamiltonian
    :param self:
    :param t:the interaction parameter
    :param v:the onsite potential
    :param u:
    :param size: Size of the lattice
    :return:
    '''

    # t = -1.0  # the interaction parameter
    # v = 0.0  # the onsite potential
    # u = 5.0
    # size = (2, 2, 2)

    boundary_condition = (
        BoundaryCondition.PERIODIC,
        BoundaryCondition.PERIODIC,
        BoundaryCondition.PERIODIC,
    )
    cubic_lattice = HyperCubicLattice(size=size, boundary_condition=boundary_condition)

    QiskitNatureSettings.use_pauli_sum_op = False

    fhm_cubic = FermiHubbardModel(
        cubic_lattice.uniform_parameters(
            uniform_interaction=t,
            uniform_onsite_potential=v,
        ),
        onsite_interaction=u,
    )

    ham = fhm_cubic.second_q_op()
    # print(ham)
    return ham


###


###
### Embed the Hamiltonian
###
### What is this?
# lmp = LatticeModelProblem(fhm_cubic)


def classical_solver(mapped_hamiltonian):
    numpy_solver = NumPyMinimumEigensolver()
    result = numpy_solver.compute_minimum_eigenvalue(operator=PauliSumOp(mapped_hamiltonian))
    ref_value = result.eigenvalue
    print(f"Reference value: {ref_value:.5f}")
    return ref_value


# Helper functions for classical shadow (see https://pennylane.ai/qml/demos/tutorial_classical_shadows)


def calculate_classical_shadow(circuit_template, params, shadow_size, num_qubits):
    """
    Given a circuit, creates a collection of snapshots consisting of a bit string
    and the index of a unitary operation.

    Args:
        circuit_template (function): A Pennylane QNode.
        params (array): Circuit parameters.
        shadow_size (int): The number of snapshots in the shadow.
        num_qubits (int): The number of qubits in the circuit.

    Returns:
        Tuple of two numpy arrays. The first array contains measurement outcomes (-1, 1)
        while the second array contains the index for the sampled Pauli's (0,1,2=X,Y,Z).
        Each row of the arrays corresponds to a distinct snapshot or sample while each
        column corresponds to a different qubit.
    """
    # applying the single-qubit Clifford circuit is equivalent to measuring a Pauli
    unitary_ensemble = [qml.PauliX, qml.PauliY, qml.PauliZ]

    # sample random Pauli measurements uniformly, where 0,1,2 = X,Y,Z
    unitary_ids = np.random.randint(0, 3, size=(shadow_size, num_qubits))
    outcomes = np.zeros((shadow_size, num_qubits))

    for ns in range(shadow_size):
        # for each snapshot, add a random Pauli observable at each location
        obs = [unitary_ensemble[int(unitary_ids[ns, i])](i) for i in range(num_qubits)]
        outcomes[ns, :] = circuit_template(params, observable=obs)

    # combine the computational basis outcomes and the sampled unitaries
    return (outcomes, unitary_ids)


def snapshot_state(b_list, obs_list):
    """
    Helper function for `shadow_state_reconstruction` that reconstructs
     a state from a single snapshot in a shadow.

    Implements Eq. (S44) from https://arxiv.org/pdf/2002.08953.pdf

    Args:
        b_list (array): The list of classical outcomes for the snapshot.
        obs_list (array): Indices for the applied Pauli measurement.

    Returns:
        Numpy array with the reconstructed snapshot.
    """
    num_qubits = len(b_list)

    # computational basis states
    zero_state = np.array([[1, 0], [0, 0]])
    one_state = np.array([[0, 0], [0, 1]])

    # local qubit unitaries
    phase_z = np.array([[1, 0], [0, -1j]], dtype=complex)
    hadamard = qml.matrix(qml.Hadamard(0))
    identity = qml.matrix(qml.Identity(0))

    # undo the rotations that were added implicitly to the circuit for the Pauli measurements
    unitaries = [hadamard, hadamard @ phase_z, identity]

    # reconstructing the snapshot state from local Pauli measurements
    rho_snapshot = [1]
    for i in range(num_qubits):
        state = zero_state if b_list[i] == 1 else one_state
        U = unitaries[int(obs_list[i])]

        # applying Eq. (S44)
        local_rho = 3 * (U.conj().T @ state @ U) - identity
        rho_snapshot = np.kron(rho_snapshot, local_rho)

    return rho_snapshot


def shadow_state_reconstruction(shadow):
    """
    Reconstruct a state approximation as an average over all snapshots in the shadow.

    Args:
        shadow (tuple): A shadow tuple obtained from `calculate_classical_shadow`.

    Returns:
        Numpy array with the reconstructed quantum state.
    """
    num_snapshots, num_qubits = shadow[0].shape

    # classical values
    b_lists, obs_lists = shadow

    # Averaging over snapshot states.
    shadow_rho = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex)
    for i in range(num_snapshots):
        shadow_rho += snapshot_state(b_lists[i], obs_lists[i])

    return shadow_rho / num_snapshots


def main():
    seed = np.random.seed(seed=int(time.time()))
    hamiltonian_jw = JordanWignerMapper().map(cubic_FHM(t=-args.hubbard_t, v=args.hubbard_v, u=args.hubbard_u, size=(
        args.hubbard_size[0], args.hubbard_size[1], args.hubbard_size[2])))
    if args.verbose > 1:
        print(hamiltonian_jw)
    reference_eigenvalue = classical_solver(hamiltonian_jw)

    ###
    ### VQE See Peruzzo, A., et al, “A variational eigenvalue solver on a quantum processor” arXiv:1304.3061
    ###

    ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")
    counts = []
    values = []
    algorithm_globals.random_seed = seed

    def store_intermediate_result(eval_count, parameters, mean, std):
        counts.append(eval_count)
        values.append(mean)

    noiseless_estimator = AerEstimator(
        run_options={"seed": seed, "shots": 1024},
        transpile_options={"seed_transpiler": seed},
        backend_options={"method": "automatic"}
    )
    if args.vqe_optimizer=='spsa':
        optimizer=SPSA(maxiter=args.vqe_maxsteps)
    if args.vqe_optimizer == 'slsqp':
        optimizer = SLSQP(maxiter=args.vqe_maxsteps)
    vqe = VQE(
        noiseless_estimator, ansatz, optimizer=optimizer, callback=store_intermediate_result
    )
    result = vqe.compute_minimum_eigenvalue(operator=hamiltonian_jw)

    print(f"VQE on Aer qasm simulator (no noise): {result.eigenvalue:.5f}")
    print(
        f"Delta from reference energy value is {abs(result.eigenvalue - reference_eigenvalue):.5f}"
    )
    import matplotlib.pyplot as plt

    plt.plot(counts, values)
    plt.xlabel("Eval count")
    plt.ylabel("Energy")
    plt.title("Convergence with no noise")

    plt.show()

    ##
    ## Now we move to pennylane for classical shadows
    ##
    # First we need to convert the optimized circuit from Qiskit VQE to a Pennylane qnode
    dev = qml.device('default.qubit', wires=hamiltonian_jw.num_qubits)

    @qml.qnode(dev)
    def tomography_circuit(params, **kwargs):
        observables = kwargs.pop("observable")
        qml.from_qiskit(result.optimal_circuit)
        return [qml.expval(o) for o in observables]

    # Now construct the shadow state
    num_snapshots = 1000
    params = []
    shadow = calculate_classical_shadow(
        tomography_circuit, params, num_snapshots, hamiltonian_jw.num_qubits
    )
    if args.verbose>2:
        print(shadow[0])
        print(shadow[1])
    # shadow_state = shadow_state_reconstruction(shadow)
    # print(np.round(shadow_state, decimals=6))

    ###
    ### Outro
    ###
    print(f'Number of qubits: {hamiltonian_jw.num_qubits}')
    print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fermi Hubbard Classical Shadow')
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    parser.add_argument("-ht", "--hubbard_t", type=float, default="-1.0", action="store", help="Interaction matrix")
    parser.add_argument("-hv", "--hubbard_v", type=float, default="0.0", action="store", help="Onsite parameter")
    parser.add_argument("-hu", "--hubbard_u", type=float, default="5.0", action="store", help="Onsite parameter")
    parser.add_argument("-hs", "--hubbard_size", type=int, nargs=3, default=[2, 2, 2], metavar=('nx', 'ny', 'nz'),
                        action="store", help="Size of the cubic lattice")
    parser.add_argument("-vqemax", "--vqe_maxsteps", type=int, default=200, action="store", help="VQE max steps")
    parser.add_argument("-vqeopt", "--vqe_optimizer", default='spsa', choices=['spsa', 'slsqp'], help="VQE optimizer")
    args = parser.parse_args()
    print("Qiskit Related:")
    print(Aer.backends())
    print("Fermi-Hubbard parameters:")
    print("t:", args.hubbard_t)
    print("v:", args.hubbard_v)
    print("u:", args.hubbard_u)
    print("Size of the SC lattice:", args.hubbard_size)
    print("VQE parameters:")
    print("Optimizer:",args.vqe_optimizer)
    print("max iterations:",args.vqe_maxsteps)
    main()
