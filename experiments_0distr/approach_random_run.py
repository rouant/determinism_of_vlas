import argparse
import numpy as np
import dill
# switch to no_similarity here
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
from pymoo.termination.max_eval import MaximumFunctionCallTermination
from pymoo.termination import get_termination

#=========REPAIR
class MyRepair(Repair):
    def _do(self, problem, X, **kwargs):
        #print("====REPAIR")
        xl=(0, 0) # lower boundaries
        xu=(4, 7) # upper boundaries
        for i_idx, individual in enumerate(X):
            coords_array = individual[6:]
            coords = [(coords_array[i], coords_array[i+1]) for i in range(0, len(coords_array), 2)]
            #print(f"COORDS: {coords}")
            for idx, c in enumerate(coords):
                others = [x for x in [(coords_array[i], coords_array[i+1]) for i in range(0, len(coords_array), 2)]]
                others.remove(c)
                no_area = []
                for o in others:
                    no_area.append(o) # base location
                    # locations in a square around base
                    no_area += [(o[0] + 1, o[1]), (o[0] - 1, o[1])]
                    no_area += [(o[0], o[1] + 1), (o[0], o[1] - 1)]
                    no_area += [(o[0] + 1, o[1] + 1), (o[0] + 1, o[1] - 1)]
                    no_area += [(o[0] - 1, o[1] + 1), (o[0] - 1, o[1] - 1)]
                while c in no_area:
                    #print("REPAIRING:", c, " FOR ", coords)
                    move = 1 if np.random.rand() < 0.5 else -1
                    axis = 0 if np.random.rand() < 0.5 else 1
                    coords_array[idx*2 + axis] += move
                    # forcing to stay within boundaries
                    if coords_array[idx*2] < xl[0]:
                        coords_array[idx*2] = xu[0] + coords_array[idx*2] + 1
                    if coords_array[idx*2 + 1] < xl[1]:
                        coords_array[idx*2 + 1] = xu[1] + coords_array[idx*2 + 1] + 1
                    if coords_array[idx*2] > xu[0]:
                        coords_array[idx*2] = coords_array[idx*2] - xu[0] - 1
                    if coords_array[idx*2 + 1] > xu[1]:
                        coords_array[idx*2 + 1] = coords_array[idx*2 + 1] - xu[1] - 1
                    c = (coords_array[idx*2], coords_array[idx*2 + 1])
            individual[6:] = coords_array
            X[i_idx] = individual
        return X


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Random runs")
    parser.add_argument('-s', '--seed', type=int, default=None, help="Random Seed")
    parser.add_argument('-m', '--model', type=str,
                        choices=["openvla", "gr00t", "spatialvla", "pi0"],
                        help="VLA model")

    args = parser.parse_args()

    SEED = args.seed # 4200
    MODEL_NAME = args.model
    # spatialvla, pi0, openvla, gr00t
    PROBLEM_NAME = f"approach_random_{MODEL_NAME}_seed-{SEED}"
    IDENTIFIER = datetime.today().strftime('%m-%d_%H-%M')
    DIVERSITY = 3 # 1 for no DIVERSITY, 3 by default
    
    GEN_SIZE = 1
    POP_STEP = 150
    POP_SIZE = 600

    np.random.seed(SEED)
    
    if MODEL_NAME == 'spatialvla':
        problem = SpatialVLAProblem(PROBLEM_NAME, DIVERSITY, IDENTIFIER)
    elif MODEL_NAME == 'pi0':
        problem = Pi0Problem(PROBLEM_NAME, DIVERSITY, IDENTIFIER)
    elif MODEL_NAME == 'openvla':
        problem = OpenVLAProblem(PROBLEM_NAME, DIVERSITY, IDENTIFIER)
    elif MODEL_NAME == 'gr00t':
        problem = Gr00tProblem(PROBLEM_NAME, DIVERSITY, IDENTIFIER)
    else:
        raise NotImplementedError
        
    #=========SAMPLING
    generated_shapes_src = np.random.randint(0, 5, size=(1, POP_SIZE))
    generated_shapes_tgt = np.random.randint(0, 2, size=(1, POP_SIZE))
    generated_nb = np.random.randint(0, 2, size=(4, POP_SIZE))
    generated_xs = np.random.randint(0, 5, size=(6, POP_SIZE))
    generated_ys = np.random.randint(0, 8, size=(6, POP_SIZE))
    generated_coords = np.empty((12, POP_SIZE), dtype=generated_xs.dtype)
    generated_coords[0::2] = generated_xs
    generated_coords[1::2] = generated_ys
    X = np.concatenate((generated_shapes_src, generated_shapes_tgt, generated_nb, generated_coords)).T
    print(X.shape)

    #=========REPAIR ON SAMPLING
    xl=(0, 0) # lower boundaries
    xu=(4, 7) # upper boundaries
    for i_idx, individual in enumerate(X):
        coords_array = individual[6:]
        coords = [(coords_array[i], coords_array[i+1]) for i in range(0, len(coords_array), 2)]
        #print(f"COORDS: {coords}")
        for idx, c in enumerate(coords):
            others = [x for x in [(coords_array[i], coords_array[i+1]) for i in range(0, len(coords_array), 2)]]
            others.remove(c)
            no_area = []
            for o in others:
                no_area.append(o) # base location
                # locations in a square around base
                no_area += [(o[0] + 1, o[1]), (o[0] - 1, o[1])]
                no_area += [(o[0], o[1] + 1), (o[0], o[1] - 1)]
                no_area += [(o[0] + 1, o[1] + 1), (o[0] + 1, o[1] - 1)]
                no_area += [(o[0] - 1, o[1] + 1), (o[0] - 1, o[1] - 1)]
            while c in no_area:
                #print("REPAIRING:", c, " FOR ", coords)
                move = 1 if np.random.rand() < 0.5 else -1
                axis = 0 if np.random.rand() < 0.5 else 1
                coords_array[idx*2 + axis] += move
                # forcing to stay within boundaries
                if coords_array[idx*2] < xl[0]:
                    coords_array[idx*2] = xu[0] + coords_array[idx*2] + 1
                if coords_array[idx*2 + 1] < xl[1]:
                    coords_array[idx*2 + 1] = xu[1] + coords_array[idx*2 + 1] + 1
                if coords_array[idx*2] > xu[0]:
                    coords_array[idx*2] = coords_array[idx*2] - xu[0] - 1
                if coords_array[idx*2 + 1] > xu[1]:
                    coords_array[idx*2 + 1] = coords_array[idx*2 + 1] - xu[1] - 1
                c = (coords_array[idx*2], coords_array[idx*2 + 1])
        individual[6:] = coords_array
        X[i_idx] = individual
    
    pbar = tqdm(total=POP_SIZE*GEN_SIZE, desc="RANDOM individuals running")
    problem.set_pbar_search(pbar)

    INPUT = X[:POP_STEP]
    NEXT_INPUT = X[POP_STEP:]
    
    problem.evaluate(INPUT)

    with open(f"approach_random_{MODEL_NAME}_seed-{SEED}.settings", "wb") as f:
        dill.dump((POP_STEP, POP_STEP, PROBLEM_NAME, IDENTIFIER, DIVERSITY, MODEL_NAME, GEN_SIZE, POP_SIZE, NEXT_INPUT, SEED, X), f)

    pbar.close()
