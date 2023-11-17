from qiskit_nature.second_q.operators import FermionicOp
from qiskit.quantum_info import SparsePauliOp,Pauli

def convert_pauli_string(ps):
    term = []

    for qubit,pauli in enumerate(ps):
        if pauli.to_label() != 'I':
            term.append((qubit,pauli))
    return term

def reverse_jwt(ops: SparsePauliOp):
    '''
    Z_j -> I - 2 a^\dagger_j a_j
    X_j -> (a^\dagger_j + a_j) Z_{j-1} Z_{j-2} .. Z_0
    Y_j -> i (a^\dagger_j - a_j) Z_{j-1} Z_{j-2} .. Z_0
    '''


    transformed_operator = FermionicOp({})

    num_qubits = ops.num_qubits
    for coeff,term_ in zip(ops.coeffs,ops.paulis):
        

        transformed_term = FermionicOp.one()

        term = convert_pauli_string(term_)
        
        if term:

            working_term = term_

            pauli_operator = term[-1]


            while pauli_operator is not None:
                
                pauli_label = pauli_operator[1].to_label()
                current_qubit = pauli_operator[0]

                if pauli_label == "Z":
                    transformed_pauli = FermionicOp.one() + FermionicOp({f"+_{current_qubit} -_{current_qubit}":-2},num_spin_orbitals=num_qubits)
                else:
                    raising_term = FermionicOp({f"+_{current_qubit}":1},num_spin_orbitals=num_qubits)
                    lovering_term = FermionicOp({f"-_{current_qubit}":1},num_spin_orbitals=num_qubits)

                    if pauli_label == "Y":
                        raising_term *= 1j
                        lovering_term *= -1j
                    
                    transformed_pauli = raising_term + lovering_term


                    for j in reversed(range(current_qubit)):

                        z_term = Pauli("I"*(num_qubits-j-1)+  "Z" + "I"*(j))
                        
                        working_term = z_term @ working_term


                    transformed_pauli *= (-1j)**(working_term.phase)

                    working_term.phase = 0


                working_qubit = current_qubit - 1
                for working_operator in reversed(convert_pauli_string(working_term)):
                    if working_operator[0] <= working_qubit:
                        pauli_operator = working_operator
                        break
                    else:
                        pauli_operator = None


                transformed_term @= transformed_pauli

        transformed_operator += coeff * transformed_term

    return transformed_operator.simplify()
