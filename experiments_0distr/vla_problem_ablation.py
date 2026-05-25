import numpy as np
from vla_instance_ablation import VLAInstance
from pymoo.core.problem import ElementwiseProblem
from getOptions import getOptions
from tqdm import tqdm

from pathlib import Path
PACKAGE_DIR = Path(__file__).parent.resolve()

class GenericProblem(ElementwiseProblem):
    def __init__(self, problem_name, model_name, variability, identifier, degree):
        super().__init__(n_var=2 + 4 + 2+2 + 2*4, # shape number + boolean distractors + xy source and target + distractors (4 max) = 18
                         n_obj=2, # metric to minimize
                         n_ieq_constr=1, # out["G"] <= 0
                         xl=[0]*18, # lower boundaries
                         xu=[4] + [1] + [1]*4 + [4, 7]*2 + [4, 7]*4, # upper boundaries
                        )  
        self.problem_name = problem_name
        self.degree = degree
        self.variability = variability
        self.model_name = model_name
        self.identifier = identifier
        self.pbar_search = None
        self.result_dir = str(PACKAGE_DIR) + "/../results/" + self.problem_name + "_" + self.model_name + "_" + self.identifier
        self.vlas = [VLAInstance(problem_name=self.problem_name, 
                                 model_name=self.model_name,
                                 identifier=self.identifier,
                                 run_number=str(nb),
                                 degree=self.degree) for nb in range(0, self.variability)]


    def no_same_location(self, xs, ys):
        result = True
        zipped = list(zip(xs, ys))
        for idx in range(len(xs)):
            coords = (xs[idx], ys[idx])
            truth = coords not in zipped[idx+1:len(xs)]
            result = result and truth
        return result

    def no_same_area(self, xs, ys):
        result = True
        coords = list(zip(xs, ys))
        for c in coords:
            others = [x for x in coords]
            others.remove(c)
            no_area = []
            for o in others:
                no_area.append(o) # base location
                # locations in a square around base
                no_area += [(o[0] + 1, o[1]), (o[0] - 1, o[1])]
                no_area += [(o[0], o[1] + 1), (o[0], o[1] - 1)]
                no_area += [(o[0] + 1, o[1] + 1), (o[0] + 1, o[1] - 1)]
                no_area += [(o[0] - 1, o[1] + 1), (o[0] - 1, o[1] - 1)]
            truth = c not in no_area
            result = result and truth
        return result
    
    def discretize(self, x):
        # as the table is longer, x axis is split in less intervals for bigger space between coords
        xs = np.array(x[0::2])
        ys = np.array(x[1::2])
        xs = -0.27 + (xs * ( (-0.03 - -0.27) / 5) )
        ys = -0.2 + (ys * ( (0.2 - -0.2) / 8) )
        return xs, ys


    def set_pbar_search(self, pbar_search):
        self.pbar_search = pbar_search

    
    def set_start_idx(self, idx):
        [vla.set_index(idx) for vla in self.vlas]


    def get_flag_diff(self, flags):
        keys = flags[0].keys()
        diff = 0
        for key in keys:
            all_same = all(f[key] == flags[0][key] for f in flags)
            diff += not all_same
        return diff

    
    def _evaluate(self, x, out, *args, **kwargs):
        out["G"] = not self.no_same_area(np.array(x[6::2]), np.array(x[7::2]))
        if out["G"]:
            [vla.set_index(int(vla.idx)+1) for vla in self.vlas]
            out["F"] = [99, 99]
        else:
            shapes = x[0:2]
            distractors = x[2:6]
            xs, ys = self.discretize(x[6:])
            values = [0]*len(x)
            values[0:2] = shapes
            values[2:6] = distractors
            values[6::2] = xs
            values[7::2] = ys
            # print(f"values: {values}")
            options, _, _ = getOptions(shapes[0], shapes[1], distractors, xs, ys)
            scores = []
            all_actions = []
            all_flags = []
            for vla in tqdm(self.vlas, desc="Running for instances"):
                vla.set_preset(options)
                score, actions, flags = vla.run_episode(values)
                scores.append(score)
                all_actions.append(actions)
                all_flags.append(flags)
                
            # get uncertainty from actions
            nb_steps = len(all_actions[0])
            variab = []
            for step in range(nb_steps):
                world_vectors = []
                rot_axangles = []
                grippers = []
                for idx in range(self.variability):
                    world_vectors.append(all_actions[idx][step]["world_vector"])
                    rot_axangles.append(all_actions[idx][step]["rot_axangle"])
                    grippers.append(all_actions[idx][step]["gripper"])
                variab.append(np.concatenate([
                    np.std(np.array(world_vectors), axis=0),
                    np.std(np.array(rot_axangles), axis=0),
                    [np.std(np.array(grippers))]
                ]))
            
            score = sorted(scores)[len(scores)//2]
            
            uncert_score = np.mean([np.mean(arr) for arr in variab])
            uncert_from_flag_diff = self.get_flag_diff(all_flags)

            #print(f"scores: {scores}")
            #print(f"mean_variab: {[np.mean(arr) for arr in variab]}")
            #print(f"uncert_score: {uncert_score}")

            file_path = self.result_dir + f"/uncert_score.csv"
            with open(file_path, "a") as f:
                f.write(str(uncert_from_flag_diff) + "\n")

            self.pbar_search.update(1)
            distr_score = sum(distractors)
            out["F"] = [score, distr_score]
                


class SpatialVLAProblem(GenericProblem):
    def __init__(self, problem_name, variability, identifier, degree):
        super().__init__(problem_name, "spatialvla-4b", variability, identifier, degree)


class Pi0Problem(GenericProblem):
    def __init__(self, problem_name, variability, identifier, degree):
        super().__init__(problem_name, "pi0", variability, identifier, degree)


class OpenVLAProblem(GenericProblem):
    def __init__(self, problem_name, variability, identifier, degree):
        super().__init__(problem_name, "openvla-7b", variability, identifier, degree)


class Gr00tProblem(GenericProblem):
    def __init__(self, problem_name, variability, identifier, degree):
        super().__init__(problem_name, "gr00t", variability, identifier, degree)


