import numpy as np
import dill
# switch to no_variability here
from vla_problem_no_similarity import SpatialVLAProblem, Pi0Problem, OpenVLAProblem, Gr00tProblem
from datetime import datetime
from tqdm import tqdm
import warnings

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from custom_sbx_no_split import SBX
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.mutation.pm import PM
from pymoo.core.repair import Repair


if __name__ == '__main__':
    with open("random_settings", 'rb') as f:
        CURRENT_POP, POP_STEP, PROBLEM_NAME, IDENTIFIER, SIMILARITY, MODEL_NAME, GEN_SIZE, POP_SIZE, PREVIOUS_INPUT = dill.load(f)

    print(f"CURRENT_POP: {CURRENT_POP}")
    
    if MODEL_NAME == 'spatialvla':
        problem = SpatialVLAProblem(PROBLEM_NAME, SIMILARITY, IDENTIFIER)
    elif MODEL_NAME == 'pi0':
        problem = Pi0Problem(PROBLEM_NAME, SIMILARITY, IDENTIFIER)
    elif MODEL_NAME == 'openvla':
        problem = OpenVLAProblem(PROBLEM_NAME, SIMILARITY, IDENTIFIER)
    elif MODEL_NAME == 'gr00t':
        problem = Gr00tProblem(PROBLEM_NAME, SIMILARITY, IDENTIFIER)
    else:
        raise NotImplementedError
        
    np.random.seed(42)  
    problem.set_start_idx(CURRENT_POP)
    
    pbar = tqdm(total=POP_SIZE*GEN_SIZE, desc="RANDOM individuals running")
    pbar.update(CURRENT_POP)
    problem.set_pbar_search(pbar)

    INPUT = PREVIOUS_INPUT[:POP_STEP]
    NEXT_INPUT = PREVIOUS_INPUT[POP_STEP:]
    
    problem.evaluate(INPUT)

    with open("random_settings", "wb") as f:
        dill.dump((CURRENT_POP+POP_STEP, POP_STEP, PROBLEM_NAME, IDENTIFIER, SIMILARITY, MODEL_NAME, GEN_SIZE, POP_SIZE, NEXT_INPUT), f)

    pbar.close()
