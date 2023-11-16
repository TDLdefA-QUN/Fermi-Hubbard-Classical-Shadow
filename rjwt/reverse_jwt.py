from qiskit_nature.second_q.operators import FermionicOp
from qiskit.quantum_info import SparsePauliOp,Pauli


def reverse_jwt(ops: SparsePauliOp):
    '''
    Z_j -> I - 2 a^\dagger_j a_j
    X_j -> (a^\dagger_j + a_j) Z_{j-1} Z_{j-2} .. Z_0
    Y_j -> i (a^\dagger_j - a_j) Z_{j-1} Z_{j-2} .. Z_0
    '''


    transformed_operator = FermionicOp({})

    num_qubits = ops.num_qubits
    for term,coeff in zip(ops.paulis,ops.coeffs):
        
        transformed_term = FermionicOp.one()
        
        working_term = term

        pauli_operator = term[-1]

        current_qubit = list(range(len(term)))[-1]

        while pauli_operator is not None:
            pauli_label = pauli_operator.to_label()

            if pauli_label == "Z":
                transformed_pauli = FermionicOp.one() + FermionicOp({f"+_{current_qubit} -_{current_qubit}":-2},num_spin_orbitals=num_qubits)


            elif pauli_label in ["X","Y"]:
                raising_term = FermionicOp({f"+_{current_qubit}":1},num_spin_orbitals=num_qubits)
                lovering_term = FermionicOp({f"-_{current_qubit}":1},num_spin_orbitals=num_qubits)

                if pauli_label == "Y":
                    raising_term *= 1j
                    lovering_term *= -1j
                
                transformed_pauli = raising_term + lovering_term

                for i in reversed(range(current_qubit)):

                    z_term = Pauli("I"*(num_qubits-i-1)+  "Z" + "I"*(i))
                    working_term = z_term @ working_term

                transformed_pauli *= (-1j)**(working_term.phase)

                working_term.phase = 0

            elif pauli_label == "I":
                transformed_pauli = FermionicOp.one()

            working_qubit = current_qubit - 1
            if working_qubit > 0:
                for working_operator_qubit,working_operator in reversed(list(enumerate(working_term))):
                    if working_operator.to_label() == "I":
                        continue

                    if working_operator_qubit <= working_qubit:
                        pauli_operator = working_operator
                        current_qubit = working_operator_qubit
                        break
                    else:
                        pauli_operator = None
            else:
                pauli_operator = None


            transformed_term @= transformed_pauli

        transformed_operator += coeff * transformed_term

    return transformed_operator.simplify()
