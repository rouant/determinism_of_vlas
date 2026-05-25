import numpy as np
from model_interface_allMetrics_custom import VLAInterface
from pathlib import Path
from tqdm import tqdm
import json
import os
import cv2
import torch
import subprocess
import random
from getOptions import getOptions, sources_map, targets_map

#torch.backends.cuda.enable_mem_efficient_sdp(False)
#torch.backends.cuda.enable_flash_sdp(False)

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Setup paths
PACKAGE_DIR = Path(__file__).parent.resolve()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

class StableJSONizer(json.JSONEncoder):
    def default(self, obj):
        return super().encode(bool(obj)) \
            if isinstance(obj, np.bool_) \
            else super().default(obj)
    
def convert_to_native(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):  
        return obj.item()
    elif isinstance(obj, list):
        return [convert_to_native(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    return obj


class VLAInstance():
    def __init__(self, problem_name, model_name, identifier, run_number):
        self.problem_name = problem_name
        self.model_name = model_name
        self.run_number = run_number
        self.identifier = identifier
        self.preset_options = {}
        self.random_seed = 42
        self.idx = "0"
        self.task_type = "put-on"
        instability_methods=['position_instability','velocity_instability', 'acceleration_instability']

        self.vla = VLAInterface(model_name=self.model_name, task="widowx_put_on_customizable",  instability_methods=instability_methods, run_number=self.run_number)

        self.result_dir = str(PACKAGE_DIR) + "/../results/" + self.problem_name + "_" + self.model_name + "_" + self.identifier + "/run-" + self.run_number
        os.makedirs(self.result_dir, exist_ok=True)


    def set_preset(self, options):
        self.preset_options = options

    
    def set_index(self, index):
        self.idx = str(int(index))


    def get_preset(self):
        source = sources_map[self.preset_options["model_ids"][0]]
        target = targets_map[self.preset_options["model_ids"][1]]
        return self.preset_options, source, target

        
    def get_score_gutscoring(self, flags):
        score = 0
        score += 4 if flags["most_moved_is_correct"] else 0
        score += 3 if flags["moved_correct_obj"] else 0
        score += -4 if flags["most_moved_is_wrong"] else 0
        score += -3 if flags["moved_wrong_obj"] else 0
        score += -1 if flags["bumped_wrong_obj"] else 0
        score += -6 if flags["moved_target"] else 0
        score += 3 if flags["correct_obj_grasped"] else 0
        score += 2 if flags["continuous_grasp"] else 0
        score += 4 if flags["correct_on_target"] else 0
        score += -2 if flags["other_on_target"] else 0
        score += -4 if flags["any_fell_off"] else 0
        score += -11 if flags["no_obj_moved"] else 0
        return score

    def get_score_73(self, flags):
        score = 0
    
        if flags["other_on_target"] and not flags["moved_correct_obj"]:
            score += 11
        elif flags["other_on_target"] and flags["moved_correct_obj"]:
            score += 23
        elif (not flags["other_on_target"] and 
                not flags["correct_on_target"] and 
                not flags["no_obj_moved"] and
                not flags["most_moved_is_correct"]):
            score += 35
        elif flags["no_obj_moved"]:
            score += 36
        elif (not flags["other_on_target"] and 
                not flags["correct_on_target"] and 
                flags["most_moved_is_correct"]):
            score += 48
        elif flags["correct_on_target"] and flags["moved_wrong_obj"]:
            score += 60
        elif flags["correct_on_target"] and not flags["moved_wrong_obj"]:
            score += 72
    
        if not flags["correct_obj_grasped"] and not flags["no_obj_moved"]:
            score += -8
        elif flags["correct_obj_grasped"] and not flags["continuous_grasp"] and not flags["no_obj_moved"]:
            score += -4
    
        # both any_fell_off and moved_target is -3 as intended
        score += -2 if flags["any_fell_off"] else 0
        score += -1 if flags["moved_target"] else 0
        
        return score

    def get_score_plus_minus(self, flags):
        score = 0
        score += 1 if flags["most_moved_is_correct"] else 0
        score += 1 if flags["moved_correct_obj"] else 0
        score += 1 if flags["bumped_correct_obj"] else 0
        score += -1 if flags["most_moved_is_wrong"] else 0
        score += -1 if flags["moved_wrong_obj"] else 0
        score += -1 if flags["bumped_wrong_obj"] else 0
        score += -1 if flags["moved_target"] else 0
        score += 1 if flags["correct_obj_grasped"] else 0
        score += 1 if flags["continuous_grasp"] else 0
        score += 1 if flags["correct_on_target"] else 0
        score += -1 if flags["other_on_target"] else 0
        score += -1 if flags["any_fell_off"] else 0
        score += -1 if flags["no_obj_moved"] else 0
        return score
        
    
    def run_episode(self, values):
        idx = self.idx
        shapes = values[0:2]
        distractors = values[2:6]
        xs = values[6::2]
        ys = values[7::2]
        if self.preset_options:
            options, source_str, target_str = self.get_preset()
        else:
            options, source_str, target_str = getOptions(shapes[0], shapes[1], distractors, xs, ys)
        prompt = f"put the {source_str} on the {target_str}"

        images, episode_stats, actions, tcp_poses, uncertainty_token, uncertainty_variability, optimal_traj, traj_inst_gradients,  traj_instability, traj_instability_tcp, exec_times_dict, gripper_distances, frame0_coords, obj_distances, episode_coords  = self.vla.run_interface(seed=self.random_seed, options=options, task_type=self.task_type, prompt=prompt)
        os.makedirs(self.result_dir + f"/{idx}", exist_ok=True)

        # log prompt
        file_path = self.result_dir + f"/{idx}/" + f"prompt.txt"
        with open(file_path, "w") as f:
            f.write(prompt)
        # log options
        file_path = self.result_dir + f"/{idx}/" + f"options.json"
        with open(file_path, "w") as f:
            json.dump(options, f, indent=4)
        # log shapes
        file_path = self.result_dir + f"/{idx}/" + f"shapes.json"
        with open(file_path, "w") as f:
            json.dump(convert_to_native(shapes), f, indent=4)
            
        # UNCERTAINTY METRICS
        #Execution time
        for method_name, values in exec_times_dict.items():
            file_path = self.result_dir + f"/{idx}/" + f"exec_time_{method_name}.json"
            with open(file_path, "w") as f:
                json.dump(convert_to_native(values), f, indent=2)
        
        # Token-based metrics
        for method_name, values in uncertainty_token.items():
            file_path = self.result_dir + f"/{idx}/" + f"Token_based_{method_name}.json"
            with open(file_path, "w") as f:
                json.dump(convert_to_native(values), f, indent=2)

        # Execution Variability
        serialized_values = [v.tolist() if isinstance(v, np.ndarray) else v for v in uncertainty_variability]
        file_path = self.result_dir + f"/{idx}/" + f"Execution_Variability.json"
        with open(file_path, "w") as f:
            json.dump(serialized_values, f, indent=4)
        
        # Optimal trajectory
        optimal_traj=np.array(optimal_traj).tolist()
        uncertainty = np.diff(optimal_traj, prepend=optimal_traj[0])
        normalized_uncertainty = ((uncertainty + 1) / 2).tolist()
        with open(self.result_dir + f"/{idx}/" + '/Optimal_Trajectory.json', "w") as f:
            json.dump(normalized_uncertainty, f, indent=4)
        
        # Trajectory Instability Gradients
        serialized_values = [v.tolist() if isinstance(v, np.ndarray) else v for v in traj_inst_gradients]
        file_path = self.result_dir + f"/{idx}/" + f"Trajectory_Instability_Gradients.json"
        with open(file_path, "w") as f:
            json.dump(serialized_values, f, indent=4)
        
        # Trajectory Instability
        for method_name, values in traj_instability.items():
            serialized_values = [v.tolist() if isinstance(v, np.ndarray) else v for v in values]
            file_path = self.result_dir + f"/{idx}/" + f"{method_name}.json"
            with open(file_path, "w") as f:
                json.dump(serialized_values, f, indent=4)
        
        for method_name, values in traj_instability_tcp.items():
            serialized_values = [v.tolist() if isinstance(v, np.ndarray) else v for v in values]
            file_path = self.result_dir + f"/{idx}/" + f"TCP_{method_name}.json"
            with open(file_path, "w") as f:
                json.dump(serialized_values, f, indent=4)
        
        
        # Episode logging
        with open(self.result_dir + f"/{idx}/" + '/log.json', "w") as f:
          json.dump(episode_stats, f, cls=StableJSONizer)
        json_ready_actions=[
          {key: value.tolist() for key, value in entry.items()}
          for entry in actions
        ]
        with open(self.result_dir + f"/{idx}/" + '/actions.json', "w") as f:
          json.dump(json_ready_actions, f, indent=2)
        with open(self.result_dir + f"/{idx}/" + '/tcp_poses.json', "w") as f:
          json.dump(tcp_poses, f, indent=2)

        # CUSTOM
        with open(self.result_dir + f"/{idx}/" + '/frame0_coords.json', "w") as f:
          json.dump(frame0_coords, f, indent=2)
        with open(self.result_dir + f"/{idx}/" + '/gripper_distances.json', "w") as f:
          json.dump(gripper_distances, f, indent=2)
        with open(self.result_dir + f"/{idx}/" + '/obj_distances.json', "w") as f:
          json.dump(obj_distances, f, indent=2)        
        with open(self.result_dir + f"/{idx}/" + '/episode_coords.json', "w") as f:
          json.dump(episode_coords, f, indent=2)      
          
        # Video saving
        video_path = os.path.join(self.result_dir , f"{idx}", f"{idx}_simulation_orig.mp4")
        video_path_dest = os.path.join(self.result_dir , f"{idx}", f"{idx}_simulation.mp4")
        
        # Get frame size from the first image
        height, width = images[0].shape[:2]
        
        # Create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        fps = 10  # Set your desired FPS
        video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        imagePath=os.path.join(self.result_dir , f"{idx}","images")
        os.makedirs(imagePath)
        for img_idx in range(len(images)):
          frame = images[img_idx]
          # Ensure frame is uint8 and in BGR (OpenCV uses BGR not RGB)
          if frame.dtype != np.uint8:
              frame = (255 * (frame - frame.min()) / (frame.ptp() + 1e-8)).astype(np.uint8)
          
          if frame.shape[2] == 3:  # RGB
              frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
          
          video.write(frame)
        
        video.release()
        command = [
          "ffmpeg",
          "-i", video_path,
          "-c:v", "h264",  # Or "openh264" if needed
          video_path_dest
        ]
        
        # Run the command
        try:
          res = subprocess.run(command, check=True, stdout=open(os.devnull, 'wb'))
          #print(f"Video saved to {video_path_dest}")
        except subprocess.CalledProcessError as e:
          print("Error during conversion:", e)
        os.remove(video_path)
        #print(f"Video saved to {video_path}")
        
        # SCORE CALCULATION
        flags = episode_stats[len(episode_stats)-1]["episode_stats"]
        score = self.get_score_73(flags)
                
        # log objectives
        file_path = self.result_dir + f"/{idx}/" + f"score.csv"
        with open(file_path, "w") as f:
            f.write(str(score))
            
        file_path = self.result_dir + f"/{idx}/" + f"flags.json"
        with open(file_path, "w") as f:
            json.dump(flags, f, indent=2, cls=StableJSONizer)  
            
        file_path = self.result_dir + f"/{idx}/" + f"distractor.csv"
        with open(file_path, "w") as f:
            f.write(str(sum(distractors)))

        self.idx = str(int(self.idx) + 1)
        if self.idx == "200":
            del self.vla
            instability_methods=['position_instability','velocity_instability', 'acceleration_instability']
            self.vla = VLAInterface(model_name=self.model_name, task="widowx_put_on_customizable",  instability_methods=instability_methods, run_number=self.run_number)
            
        # OUT of FUNCTION
        return score, json_ready_actions, flags

      
