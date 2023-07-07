import qiskit.quantum_info
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import (
    BoundaryCondition,
    LineLattice,
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
from qiskit.quantum_info import DensityMatrix, state_fidelity
import psutil
import numpy as np
import matplotlib.pyplot as plt
import rustworkx as rx
from rustworkx.visualization import mpl_draw
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


def linear_FHM(t, v, u, size):
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

    boundary_condition = (
        BoundaryCondition.PERIODIC,
    )
    linear_lattice = LineLattice(num_nodes=size, boundary_condition=boundary_condition)
    # linear_lattice.draw()
    QiskitNatureSettings.use_pauli_sum_op = False

    fhm_linear = FermiHubbardModel(
        linear_lattice.uniform_parameters(
            uniform_interaction=t,
            uniform_onsite_potential=v,
        ),
        onsite_interaction=u,
    )

    ham = fhm_linear.second_q_op()
    # print(ham)
    return ham


###

###
### Embed the Hamiltonian
###

def classical_solver(mapped_hamiltonian):
    numpy_solver = NumPyMinimumEigensolver()
    result = numpy_solver.compute_minimum_eigenvalue(operator=PauliSumOp(mapped_hamiltonian))
    ref_value = result.eigenvalue
    print(f"Reference value: {ref_value:.5f}")
    return ref_value


# Helper functions for classical shadow See
# (https://github.com/ryanlevy/shadow-tutorial/blob/main/Tutorial_Shadow_State_Tomography.ipynb)

###
### Classical shadow with random clifford circuits
###
def Minv(N, X):
    '''inverse shadow channel'''
    return (2 ** N + 1.) * X - np.eye(2 ** N)


def CS_clifford(nshadows, reps, Nqubits, circuit, seed):
    '''
    Classical shadow with random Clifford circuits
    Args:
        nshadows: Number of shadows
        reps: Repetitions
        Nqubits: Number of qubits
        Circuit: Circuit to calculate the classical shadow of.
        seed: Random seed
    Returns: rho_shadow, the density matrix of the classical shadow

    '''
    rng = np.random.default_rng(seed)
    cliffords = [qiskit.quantum_info.random_clifford(Nqubits, seed=rng) for _ in range(nshadows)]

    results = []
    for cliff in cliffords:
        qc_c = circuit.compose(cliff.to_circuit())
        counts = qiskit.quantum_info.Statevector(qc_c).sample_counts(reps)
        results.append(counts)
    shadows = []
    for cliff, res in zip(cliffords, results):
        mat = cliff.adjoint().to_matrix()
        for bit, count in res.items():
            Ub = mat[:, int(bit, 2)]  # this is Udag|b>
            shadows.append(Minv(Nqubits, np.outer(Ub, Ub.conj())) * count)

    rho_shadow = np.sum(shadows, axis=0) / (nshadows * reps)
    return DensityMatrix(rho_shadow)


###
### Classical shadow with Paulis
###
def rotGate(g):
    '''produces gate U such that U|psi> is in Pauli basis g'''
    if g == "X":
        return 1 / np.sqrt(2) * np.array([[1., 1.], [1., -1.]])
    elif g == "Y":
        return 1 / np.sqrt(2) * np.array([[1., -1.0j], [1., 1.j]])
    elif g == "Z":
        return np.eye(2)
    else:
        raise NotImplementedError(f"Unknown gate {g}")


def bitGateMap(qc, g, qi):
    '''Map X/Y/Z string to qiskit ops'''
    if g == "X":
        qc.h(qi)
    elif g == "Y":
        qc.sdg(qi)
        qc.h(qi)
    elif g == "Z":
        pass
    else:
        raise NotImplementedError(f"Unknown gate {g}")


def CS_paulis(nshadows, Nqubits, circuit, seed):
    '''
    Classical shadow with random Clifford circuits
    Args:
        nshadows: Number of shadows
        Nqubits: Number of qubits
        Circuit: Circuit to calculate the classical shadow of.
        seed: Random seed
    Returns: rho_shadow, the density matrix of the classical shadow

    '''
    rng = np.random.default_rng(seed)
    scheme = [rng.choice(['X', 'Y', 'Z'], size=Nqubits) for _ in range(nshadows)]
    labels, counts = np.unique(scheme, axis=0, return_counts=True)
    results = []
    for bit_string, count in zip(labels, counts):
        qc_m = circuit.copy()
        # rotate the basis for each qubit
        for i, bit in enumerate(bit_string):
            bitGateMap(qc_m, bit, i)
        counts = qiskit.quantum_info.Statevector(qc_m).sample_counts(count)
        results.append(counts)

    shadows = []
    shots = 0
    for pauli_string, counts in zip(labels, results):
        # iterate over measurements
        for bit, count in counts.items():
            mat = 1.
            for i, bi in enumerate(bit[::-1]):
                b = rotGate(pauli_string[i])[int(bi), :]
                mat = np.kron(Minv(1, np.outer(b.conj(), b)), mat)
            shadows.append(mat * count)
            shots += count

    rho_shadow = np.sum(shadows, axis=0) / shots
    return DensityMatrix(rho_shadow)


def main():
    if args.dimensions == 3:
        if len(args.hubbard_size) != 3:
            print("Size argument must exactly have three entries")
            exit()
        hamiltonian_jw = JordanWignerMapper().map(cubic_FHM(t=args.hubbard_t, v=args.hubbard_v, u=args.hubbard_u, size=(
            args.hubbard_size[0], args.hubbard_size[1], args.hubbard_size[2])))
    if args.dimensions == 1:
        if len(args.hubbard_size) != 1:
            print("Size argument must exactly have one entry")
            exit()
        hamiltonian_jw = JordanWignerMapper().map(
            linear_FHM(t=args.hubbard_t, v=args.hubbard_v, u=args.hubbard_u, size=
            args.hubbard_size[0]))
    if args.dimensions == 2:
        if len(args.hubbard_size) != 2:
            print("Size argument must exactly have two entries")
            exit()
        hamiltonian_jw = JordanWignerMapper().map(
            linear_FHM(t=args.hubbard_t, v=args.hubbard_v, u=args.hubbard_u, size=
            (args.hubbard_size[0], args.hubbard_size[1])))

    if args.verbose > 1:
        print(hamiltonian_jw)
    reference_eigenvalue = classical_solver(hamiltonian_jw)

    ###
    ### VQE See Peruzzo, A., et al, “A variational eigenvalue solver on a quantum processor” arXiv:1304.3061
    ###

    ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")
    counts = []
    values = []
    algorithm_globals.random_seed = np.random.default_rng(args.seed)

    def store_intermediate_result(eval_count, parameters, mean, std):
        if args.verbose > 1 :
            print("VQE step:",eval_count)
        counts.append(eval_count)
        values.append(mean)

    noiseless_estimator = AerEstimator(
        run_options={"seed": np.random.seed(args.seed), "shots": 1024},
        transpile_options={"seed_transpiler": np.random.seed(args.seed)},
        backend_options={"method": "automatic"}
    )
    if args.vqe_optimizer == 'spsa':
        optimizer = SPSA(maxiter=args.vqe_maxsteps)
    if args.vqe_optimizer == 'slsqp':
        optimizer = SLSQP(maxiter=args.vqe_maxsteps)
    vqe = VQE(
        noiseless_estimator, ansatz, optimizer=optimizer, callback=store_intermediate_result
    )
    result = vqe.compute_minimum_eigenvalue(operator=hamiltonian_jw)
    optimum_circuit = ansatz.bind_parameters(result.optimal_parameters)
    print(f"VQE on Aer qasm simulator (no noise): {result.eigenvalue:.5f}")
    print(f"Delta from reference energy value is {abs(result.eigenvalue - reference_eigenvalue):.5f}")
    if args.verbose:
        plt.plot(counts, values)
        plt.xlabel("Eval count")
        plt.ylabel("Energy")
        plt.title("Convergence with no noise")
        plt.savefig(
            'VQE_convergence-{}d-{}-{}-{}iter.png'.format(args.dimensions, args.hubbard_size, args.vqe_optimizer,
                                                          args.vqe_maxsteps))

    # Density matrix of the final optimized state

    if args.verbose > 1:
        print(result.optimal_parameters)

    vqe_density_matrix = DensityMatrix.from_instruction(optimum_circuit)
    # vqe_density_matrix = simulator_density_matrix(circuit=result.optimal_circuit, use_gpu=False)
    if args.verbose:
        figure = vqe_density_matrix.draw(output='hinton')
        figure.savefig('VQE_final_density_matrix-hinton-{}d-{}-{}-{}iter.png'.format(args.dimensions, args.hubbard_size,
                                                                                     args.vqe_optimizer,
                                                                                     args.vqe_maxsteps))
        text = vqe_density_matrix.draw(output='latex_source')
        f = open('VQE_final_density_matrix-latex-{}d-{}-{}-{}iter.tex'.format(args.dimensions, args.hubbard_size,
                                                                              args.vqe_optimizer,
                                                                              args.vqe_maxsteps), "w")
        f.write(text)
        f.close()
    if args.shadow_sampler == "clifford":
        print("Using random Clifford circuits to construct classical shadow")
        cs_density_matrix = CS_clifford(nshadows=args.classical_snapshots, reps=1, Nqubits=hamiltonian_jw.num_qubits,
                                        circuit=optimum_circuit, seed=args.seed)
    if args.shadow_sampler == "paulis":
        print("Using random Paulis to construct classical shadow")
        cs_density_matrix = CS_paulis(nshadows=args.classical_snapshots, Nqubits=hamiltonian_jw.num_qubits,
                                      circuit=optimum_circuit, seed=args.seed)
    if args.verbose:
        figure = cs_density_matrix.draw(output='hinton')
        figure.savefig('CS_density_matrix-hinton-{}d-{}-{}snapshots-{}.png'.format(args.dimensions, args.hubbard_size,
                                                                                   args.classical_snapshots,
                                                                                   args.shadow_sampler))
        text = cs_density_matrix.draw(output='latex_source')
        f = open('CS_density_matrix-latex-{}d-{}-{}snapshots-{}.tex'.format(args.dimensions, args.hubbard_size,
                                                                            args.classical_snapshots,
                                                                            args.shadow_sampler), "w")
        f.write(text)
        f.close()
    print("Fidelity=", qiskit.quantum_info.state_fidelity(vqe_density_matrix, cs_density_matrix, validate=False))

    ###
    ### Outro
    ###
    print(f'Number of qubits: {hamiltonian_jw.num_qubits}')
    print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fermi Hubbard Classical Shadow')
    parser.add_argument("-v", "--verbose", action="count",default=0, help="increase output verbosity")
    parser.add_argument("-ht", "--hubbard_t", type=float, default="-1.0", action="store", help="Interaction matrix")
    parser.add_argument("-hv", "--hubbard_v", type=float, default="0.0", action="store", help="Onsite parameter")
    parser.add_argument("-hu", "--hubbard_u", type=float, default="5.0", action="store", help="Onsite parameter")
    parser.add_argument("-hs", "--hubbard_size", type=int, nargs='+', default=[2],
                        action="store", help="Size of the cubic lattice")
    parser.add_argument("-vqemax", "--vqe_maxsteps", type=int, default=200, action="store", help="VQE max steps")
    parser.add_argument("-dim", "--dimensions", type=int, default=1, choices=range(1, 4), help="Periodic dimensions")
    parser.add_argument("-vqeopt", "--vqe_optimizer", default='spsa', choices=['spsa', 'slsqp'], help="VQE optimizer")
    parser.add_argument("-csc", "--shadow_sampler", default='clifford', choices=['clifford', 'paulis'],
                        help="How to calculate the classical shadow")
    parser.add_argument("-csn", "--classical_snapshots", type=int, default=1000, action="store",
                        help="Classical shadow snapshots")
    parser.add_argument("-s", "--seed", type=int, default=int(time.time()), action="store",
                        help="Random seed")
    parser.add_argument("-gpu", default=False, action="store_true",
                        help="Enable GPU")
    args = parser.parse_args()
    if args.verbose:
        print("Seed:", args.seed)
        print("Qiskit Related:")
        print(Aer.backends())
    print("Fermi-Hubbard parameters:")
    if args.dimensions == 1:
        print("Linear periodic lattice")
    if args.dimensions == 2:
        print("Square periodic lattice")
    if args.dimensions == 3:
        print("Cubic periodic lattice")
    print("Size of the lattice:", args.hubbard_size)
    print("t:", args.hubbard_t)
    print("v:", args.hubbard_v)
    print("u:", args.hubbard_u)
    print("VQE parameters:")
    print("Optimizer:", args.vqe_optimizer)
    print("max iterations:", args.vqe_maxsteps)
    print("Classical shadow related:")
    print("No snapshots:", args.classical_snapshots)
    print("")
    if args.gpu:
        print("GPU will be used")
    main()
