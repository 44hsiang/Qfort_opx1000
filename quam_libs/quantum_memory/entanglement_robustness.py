import sys
from qutip import Qobj
from picos import Problem, value
from picos.expressions.variables import HermitianVariable
from picos.expressions.algebra import trace, partial_transpose


def entanglementRobustness(state, solver='mosek', **extra_options) :
    if isinstance(state, Qobj):
        state = (state).full()
    
    SP = Problem()

    # add variable
    gamma = HermitianVariable("gamma", (4, 4))
    rho = HermitianVariable("rho", (4, 4))

    # add constraints
    SP.add_constraint(
        (state + gamma) - rho == 0
    )
    SP.add_constraint(
        partial_transpose(rho, 0) >> 0
    )
    SP.add_constraint(
        gamma >> 0
    )
    SP.add_constraint(
        trace(rho) - 1 >> 0
    )
    
    # find the solution
    SP.set_objective(
        'min',
        trace(rho) - 1
    )

    # solve the problem
    SP.solve(solver=solver, **extra_options)

    # return results
    return max(SP.value, 0)




if __name__ == "__main__":
    from qutip import Qobj
    import numpy as np


    choi_matrix = np.array([
        [0.5, 0, 0, 0.5],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0.5, 0, 0, 0.5]
    ], dtype=complex)

    rho = Qobj(choi_matrix)

    R = entanglementRobustness(rho)
    print("Entanglement robustness =", R)
