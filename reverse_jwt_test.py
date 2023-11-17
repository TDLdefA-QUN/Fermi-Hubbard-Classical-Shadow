import qiskit_nature
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from reverse_jwt import reverse_jwt

qiskit_nature.settings.use_pauli_sum_op = False

def test(op,coef):
    fo = FermionicOp({op: coef}).normal_order()
    qo = JordanWignerMapper().map(fo)
    rjwt = reverse_jwt(qo)

    return fo, rjwt, fo.equiv(rjwt)


tests = [
    "-_0",
    "-_1",
    "-_2",
    "-_3",
    
    "+_0",
    "+_1",
    "+_2",
    "+_3",

    "-_0 +_0",
    "+_0 -_0",
    "-_0 -_0",
    "+_0 +_0",

    "-_0 +_1", 
    "+_1 -_0",

    "-_0 -_1",
    "-_1 -_0",

    "+_1 +_0",
    "+_0 +_1",
    
    "-_0 +_2",
    "+_2 -_0",

    "-_0 -_2",
    "-_2 -_0",

    "+_2 +_0",
    "+_0 +_2",

    "-_1 +_2",
    "+_2 -_1",

    "-_1 -_2",
    "-_2 -_1",

    "+_2 +_1",
    "+_1 +_2",

    "-_0 -_1 -_2",
    "+_0 -_1 +_2",
    "+_0 -_1 -_2",
    "+_0 +_1 -_2",
    "-_0 +_1 -_2",
    "-_0 +_1 -_3",

    "-_0 -_1 -_2 -_3",
    

    ]


for t in tests:
    print(test(t,1),end="\n\n")