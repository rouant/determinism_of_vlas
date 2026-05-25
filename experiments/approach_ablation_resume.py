import argparse
import numpy as np
import dill
# switch to no_similarity here
from vla_problem_ablation import SpatialVLAProblem, Pi0Problem, OpenVLAProblem, Gr00tProblem
from datetime import datetime
from tqdm import tqdm
import warnings

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from custom_sbx_no_split import SBX
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.mutation.pm import PM
from pymoo.core.repair import Repair
from pymoo.termination.max_gen import MaximumGenerationTermination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Ablation runs")
    parser.add_argument('-s', '--seed', type=int, default=None, help="Random Seed")
    parser.add_argument('-d', '--degree', type=int, default=None, 
                        choices=[1, 2, 3],
                        help="Degree of ablation")
    parser.add_argument('-m', '--model', type=str,
                        choices=["openvla", "gr00t", "spatialvla", "pi0"],
                        help="VLA model")

    args = parser.parse_args()

    SEED = args.seed # 4200
    DEGREE = args.degree # 1, 2, 3
    MODEL_NAME = args.model
    
    with open(f"approach_ablation_degree-{DEGREE}_{MODEL_NAME}_seed-{SEED}_settings", 'rb') as f:
        CURRENT_GEN, GEN_STEP, PROBLEM_NAME, IDENTIFIER, DIVERSITY, MODEL_NAME, GEN_SIZE, POP_SIZE, SEED, DEGREE = dill.load(f)

    with open(f"approach_ablation_degree-{DEGREE}_{MODEL_NAME}_seed-{SEED}_checkpoint", 'rb') as f:
        checkpoint = dill.load(f)
        print("Loaded Checkpoint:", checkpoint)

    checkpoint.termination = MaximumGenerationTermination(min(CURRENT_GEN + GEN_STEP, GEN_SIZE))
    
    if MODEL_NAME == 'spatialvla':
        checkpoint.problem = SpatialVLAProblem(PROBLEM_NAME, DIVERSITY, IDENTIFIER, DEGREE)
    elif MODEL_NAME == 'pi0':
        checkpoint.problem = Pi0Problem(PROBLEM_NAME, DIVERSITY, IDENTIFIER, DEGREE)
    elif MODEL_NAME == 'openvla':
        checkpoint.problem = OpenVLAProblem(PROBLEM_NAME, DIVERSITY, IDENTIFIER, DEGREE)
    elif MODEL_NAME == 'gr00t':
        checkpoint.problem = Gr00tProblem(PROBLEM_NAME, DIVERSITY, IDENTIFIER, DEGREE)
    else:
        raise NotImplementedError

    checkpoint.problem.set_start_idx(CURRENT_GEN*POP_SIZE)
    np.random.seed(SEED)  
    
    pbar = tqdm(total=POP_SIZE*GEN_SIZE, desc="NGSA-II individuals running")
    pbar.update(CURRENT_GEN*POP_SIZE)
    checkpoint.problem.set_pbar_search(pbar)
    
    res = minimize(
                checkpoint.problem,
                checkpoint,
                seed=SEED,
                copy_algorithm=False,
                verbose=True)

    del checkpoint.problem # for pickling efficiency
    with open(f"approach_ablation_degree-{DEGREE}_{MODEL_NAME}_seed-{SEED}_checkpoint", "wb") as f:
        dill.dump(checkpoint, f)

    with open(f"approach_ablation_degree-{DEGREE}_{MODEL_NAME}_seed-{SEED}_settings", "wb") as f:
        dill.dump((CURRENT_GEN+GEN_STEP, GEN_STEP, PROBLEM_NAME, IDENTIFIER, DIVERSITY, MODEL_NAME, GEN_SIZE, POP_SIZE, SEED, DEGREE), f)

    filename = 'pareto_' + PROBLEM_NAME + '_' + MODEL_NAME + IDENTIFIER + '.txt'
    with open(filename, "a") as f:
        f.write(str(res.F))
    print(res)
    #print(res.pareto_front())
    pbar.close()
