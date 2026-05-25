from collections import OrderedDict
from typing import List

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat

from mani_skill2_real2sim.utils.common import random_choice
from mani_skill2_real2sim.utils.registration import register_env
from mani_skill2_real2sim import ASSET_DIR

from .base_env import CustomBridgeObjectsInSceneEnv, CustomOtherObjectsInSceneEnv
from .move_near_in_scene_customizable import MoveNearInSceneCustomizableEnv

from mani_skill2_real2sim.utils.sapien_utils import (
    get_actor_state,
    get_articulation_state,
    set_actor_state,
    set_articulation_state,
)


class PutOnInSceneCustomizableEnv(MoveNearInSceneCustomizableEnv):

    #CUSTOM
    def get_coordinates(self):
        coords = {}
        for actor in self._actors[1:]:
            coords[actor.name] = [float(x) for x  in get_actor_state(actor)[0:3]]
        return coords
    
    def reset(self, *args, **kwargs):
        self.continuous_grasp = 0
        self.consecutive_grip_frozen = 0
        self.previous_gripper = []
        self.previous_poses = []
        self.total_distances = []
        return super().reset(*args, **kwargs)

    def _initialize_episode_stats(self):
        self.episode_stats = OrderedDict(
            most_moved_is_correct=False,
            most_moved_is_wrong=False,
            correct_obj_grasped=False,
            continuous_grasp=False,
            correct_on_target=False,
            other_on_target=False,
            bumped_wrong_obj=False,
            moved_correct_obj=False,
            moved_wrong_obj=False,

            bumped_target=False,
            moved_target=False,
            
            no_obj_moved=False,
            gripper_froze=False,
            bumped_correct_obj=False,
        )

    def _set_model(self, model_ids, model_scales):
        """Set the model id and scale. If not provided, choose one randomly."""

        if model_ids is None:
            src_model_id = random_choice(self.model_ids, self._episode_rng)
            tgt_model_id = (self.model_ids.index(src_model_id) + 1) % len(
                self.model_ids
            )
            model_ids = [src_model_id, tgt_model_id]

        return super()._set_model(model_ids, model_scales)

    def evaluate(self, success_require_src_completely_on_target=True, z_flag_required_offset=0.02, **kwargs):
        source_obj_pose = self.source_obj_pose
        target_obj_pose = self.target_obj_pose

        # total distance moved
        if len(self.total_distances) == 0:
            # init total_distances to zero
            self.total_distances = np.zeros(len(self.episode_objs))
        if len(self.previous_poses) > 0:
            distances = [np.linalg.norm(
                previous.p[:2] - current.pose.p[:2]
            ) for current, previous in zip(self.episode_objs, self.previous_poses)]
            #print(f"distances: {distances}")
            self.total_distances += distances
        
        # distance affectation robust to objs order
        source_distance = 0
        target_distance = 0
        others_distances = []
        for idx, obj in enumerate(self.episode_objs):
            if obj.name == self.episode_source_obj.name:
                source_distance = self.total_distances[idx]
                continue
            if obj.name == self.episode_target_obj.name:
                target_distance = self.total_distances[idx]
                continue
            others_distances.append(self.total_distances[idx])
        # set for next loop
        self.previous_poses = [obj.pose for obj in self.episode_objs]
        #print(f"source, target, others distance: {source_distance}, {target_distance}, {others_distances}")

        target_obj_xy_move_dist = np.linalg.norm(
            self.episode_target_obj_xyz_after_settle[:2] - self.target_obj_pose.p[:2]
        )
        #print(f"target_obj_xy_move_dist: {target_obj_xy_move_dist}")
        
        # whether moved the correct object
        source_obj_xy_move_dist = np.linalg.norm(
            self.episode_source_obj_xyz_after_settle[:2] - self.source_obj_pose.p[:2]
        )
        other_obj_xy_move_dist = []
        for obj, obj_xyz_after_settle in zip(
                self.episode_objs, self.episode_obj_xyzs_after_settle
        ):
            if obj.name == self.episode_source_obj.name:
                continue
            if obj.name == self.episode_target_obj.name:
                continue
            other_obj_xy_move_dist.append(
                np.linalg.norm(obj_xyz_after_settle[:2] - obj.pose.p[:2])
            )
        #most_moved_is_correct = (source_obj_xy_move_dist > 0.03) and (
        #    target_obj_xy_move_dist < source_obj_xy_move_dist) and (
        #    all([x < source_obj_xy_move_dist for x in other_obj_xy_move_dist])
        #)
        most_moved_is_correct = (source_distance > 0.03) and (
            target_distance < source_distance) and (
            all([x < source_distance for x in others_distances])
        )

        # CUSTOM
        # moved_correct_obj = (source_obj_xy_move_dist > 0.03)
        # moved_wrong_obj = any([x > 0.03 for x in other_obj_xy_move_dist])
        # bumped_correct_obj = (source_obj_xy_move_dist > 0.001)
        
        # bumped_target = (target_obj_xy_move_dist > 0.001)
        # moved_target = (target_obj_xy_move_dist > 0.03)
        
        # most_moved_is_wrong = any([x > 0.03 for x in other_obj_xy_move_dist]) and any(
        #     [x > source_obj_xy_move_dist for x in other_obj_xy_move_dist]
        # )
        
        # bumped_wrong_obj = any([x > 0.001 for x in other_obj_xy_move_dist])
        # no_obj_moved = (source_obj_xy_move_dist < 0.03) and (target_obj_xy_move_dist < 0.03) and all(
        #     [x < 0.03 for x in other_obj_xy_move_dist]
        # )
        moved_correct_obj = (source_distance > 0.03)
        moved_wrong_obj = any([x > 0.03 for x in others_distances])
        bumped_correct_obj = (source_distance > 0.001)

        bumped_target = (target_distance > 0.001)
        moved_target = (target_distance > 0.03)
        
        most_moved_is_wrong = any([x > 0.03 for x in others_distances]) and any(
            [x > source_distance for x in others_distances]
        )
        
        bumped_wrong_obj = any([x > 0.001 for x in others_distances])
        no_obj_moved = (source_distance < 0.03) and (target_distance < 0.03) and all(
            [x < 0.03 for x in others_distances]
        )

        # whether any item fell off the table
        objects = self.episode_objs
        zs = [o.pose.p[2] for o in objects]
        any_fell_off = any([z < 0.8 for z in zs])

        # whether the gripper froze
        gripper_froze = False
        current_gripper_coords = self.agent.ee_pose.p # [x,y,z]
        #print(f"current_gripper_coords:{current_gripper_coords}")
        if len(self.previous_gripper) == 0:
            self.previous_gripper = current_gripper_coords
        else:
            #print(f"previous_gripper:{self.previous_gripper}")
            dist = np.linalg.norm(self.previous_gripper - current_gripper_coords)
            #print(f"dist:{dist}")
            if dist < 0.03 :
                self.consecutive_grip_frozen += 1
                if self.consecutive_grip_frozen > 10:
                    gripper_froze = True
            else:
                self.consecutive_grip_frozen = 0
        self.previous_gripper = current_gripper_coords
                    

        # whether the source object is grasped
        correct_obj_grasped = self.agent.check_grasp(self.episode_source_obj)
        if correct_obj_grasped:
            self.continuous_grasp += 1
        else:
            self.continuous_grasp = 0
        continuous_grasp = self.continuous_grasp >= 5

        # whether the source object is on the target object based on bounding box position
        tgt_obj_half_length_bbox = (
                self.episode_target_obj_bbox_world / 2
        )  # get half-length of bbox xy diagonol distance in the world frame at timestep=0
        src_obj_half_length_bbox = self.episode_source_obj_bbox_world / 2

        pos_src = source_obj_pose.p
        pos_tgt = target_obj_pose.p
        offset = pos_src - pos_tgt
        xy_flag = (
                np.linalg.norm(offset[:2])
                <= np.linalg.norm(tgt_obj_half_length_bbox[:2]) + 0.003
        )
        z_flag = (offset[2] > 0) and (
                offset[2] - tgt_obj_half_length_bbox[2] - src_obj_half_length_bbox[2]
                <= z_flag_required_offset
        )
        correct_on_target = xy_flag and z_flag

        if success_require_src_completely_on_target:
            # whether the source object is on the target object based on contact information
            contacts = self._scene.get_contacts()
            flag = True
            robot_link_names = [x.name for x in self.agent.robot.get_links()]
            tgt_obj_name = self.episode_target_obj.name
            ignore_actor_names = [tgt_obj_name] + robot_link_names
            for contact in contacts:
                actor_0, actor_1 = contact.actor0, contact.actor1
                other_obj_contact_actor_name = None
                if actor_0.name == self.episode_source_obj.name:
                    other_obj_contact_actor_name = actor_1.name
                elif actor_1.name == self.episode_source_obj.name:
                    other_obj_contact_actor_name = actor_0.name
                if other_obj_contact_actor_name is not None:
                    # the object is in contact with an actor
                    contact_impulse = np.sum(
                        [point.impulse for point in contact.points], axis=0
                    )
                    if (other_obj_contact_actor_name not in ignore_actor_names) and (
                            np.linalg.norm(contact_impulse) > 1e-6
                    ):
                        # the object has contact with an actor other than the robot link or the target object, so the object is not yet put on the target object
                        flag = False
                        break
            correct_on_target = correct_on_target and flag
        
        success = correct_on_target

        # other items on the plate
        other_on_target = False
        tgt_obj_half_length_bbox = (
                self.episode_target_obj_bbox_world / 2
        ) 
        
        for model_id, model_pose, model_bbox in zip(
                self.episode_model_ids[2:], 
            [o.pose for o in self.episode_objs][2:], 
             self.episode_model_bbox_sizes[2:]
        ):
            
            bbox_world = quat2mat(model_pose.q) @ model_bbox
            #print(f"bbox_world: {bbox_world}")
            obj_half_length_bbox = bbox_world / 2
            pos_obj = model_pose.p
            pos_tgt = target_obj_pose.p
            offset = pos_obj - pos_tgt
            xy_flag = (
                    np.linalg.norm(offset[:2])
                    <= np.linalg.norm(tgt_obj_half_length_bbox[:2]) + 0.003
            )
            z_flag = (offset[2] > 0) and (
                    offset[2] - tgt_obj_half_length_bbox[2] - obj_half_length_bbox[2]
                    <= z_flag_required_offset
            )
            #print(f"z_offset: {offset[2] - tgt_obj_half_length_bbox[2] - obj_half_length_bbox[2]}")
            #print(f"xy_offset: {np.linalg.norm(offset[:2])}")
            #print(f"xy_required_offset: {np.linalg.norm(tgt_obj_half_length_bbox[:2]) + 0.003}")
            #print(f"xy_flag: {xy_flag}")
            #print(f"z_flag_required_offset: {z_flag_required_offset}")
            other_on_target = other_on_target or (xy_flag and z_flag)
        

        self.episode_stats["most_moved_is_correct"] = most_moved_is_correct
        self.episode_stats["moved_correct_obj"] = moved_correct_obj
        self.episode_stats["bumped_correct_obj"] = (
                self.episode_stats["bumped_correct_obj"] or bumped_correct_obj
        )
        
        self.episode_stats["most_moved_is_wrong"] = most_moved_is_wrong
        self.episode_stats["moved_wrong_obj"] = moved_wrong_obj
        self.episode_stats["bumped_wrong_obj"] = (
                self.episode_stats["bumped_wrong_obj"] or bumped_wrong_obj
        )

        self.episode_stats["moved_target"] = (
                self.episode_stats["moved_target"] or moved_target
        )
        self.episode_stats["bumped_target"] = (
                self.episode_stats["bumped_target"] or bumped_target
        )

        self.episode_stats["correct_obj_grasped"] = (
                self.episode_stats["correct_obj_grasped"] or correct_obj_grasped
        )
        self.episode_stats["continuous_grasp"] = (
                self.episode_stats["continuous_grasp"] or continuous_grasp
        )
        
        self.episode_stats["correct_on_target"] = correct_on_target
        self.episode_stats["other_on_target"] = (
                self.episode_stats["other_on_target"] or other_on_target
        )

        self.episode_stats["any_fell_off"] = any_fell_off
        self.episode_stats["no_obj_moved"] = no_obj_moved
        self.episode_stats["gripper_froze"] = (
                self.episode_stats["gripper_froze"] or gripper_froze
        )
        
        

        return dict(
            most_moved_is_correct=most_moved_is_correct,
            moved_correct_obj=moved_correct_obj,
            bumped_correct_obj=bumped_correct_obj,

            most_moved_is_wrong=most_moved_is_wrong,
            moved_wrong_obj=moved_wrong_obj,
            bumped_wrong_obj=bumped_wrong_obj,

            moved_target=moved_target,
            bumped_target=bumped_target,
            
            correct_obj_grasped=correct_obj_grasped,
            continuous_grasp=continuous_grasp,
            
            correct_on_target=correct_on_target,
            other_on_target=other_on_target,

            any_fell_off=any_fell_off,
            no_obj_moved=no_obj_moved,
            gripper_froze=gripper_froze,
            
            episode_stats=self.episode_stats,
            success=success,
        )

    def get_language_instruction(self, **kwargs):
        src_name = self._get_instruction_obj_name(self.episode_source_obj.name)
        tgt_name = self._get_instruction_obj_name(self.episode_target_obj.name)
        return f"put {src_name} on {tgt_name}"


class PutOnBridgeInSceneEnv(PutOnInSceneCustomizableEnv, CustomOtherObjectsInSceneEnv):
    DEFAULT_MODEL_JSON = "info_bridge_custom_v0.json"
    def __init__(
            self,
            source_obj_name: str = None,
            target_obj_name: str = None,
            xy_configs: List[np.ndarray] = None,
            quat_configs: List[np.ndarray] = None,
            **kwargs,
    ):
        self._source_obj_name = source_obj_name
        self._target_obj_name = target_obj_name
        self._xy_configs = xy_configs
        self._quat_configs = quat_configs
        super().__init__(**kwargs)

    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "widowx"
        ret["control_freq"] = 5
        ret["sim_freq"] = 500
        ret["control_mode"] = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
        ret["scene_name"] = "bridge_table_1_v1"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/bridge_real_eval_1.png"
        )
        ret["rgb_overlay_cameras"] = ["3rd_view_camera"]

        return ret

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        self.set_episode_rng(seed)

        if "model_ids" in options.keys():
            model_ids = options.get("model_ids", [])
            obj_init_options = options.get("obj_init_options", {})
            obj_init_options = obj_init_options.copy()
            init_xys = obj_init_options.get("init_xys", [])
            init_rot_quats = obj_init_options.get("init_rot_quats", [])
            if len(init_xys) != len(model_ids) or len(init_rot_quats) != len(model_ids):
                init_xys = []
                init_rot_quats = []
                for model in model_ids:
                    per_model_init_options = obj_init_options.get(model, {})
                    obj_init_xy = per_model_init_options.get("init_xy", None)
                    if not obj_init_xy:
                        obj_init_xy = self._episode_rng.uniform([-0.3, -0.16], [0, 0.12], [2])
                    init_xys.append(obj_init_xy)
                    obj_init_rot_quat = per_model_init_options.get("init_rot_quat", None)
                    if not obj_init_rot_quat:
                        obj_init_rot_quat = euler2quat(np.pi / 2, 0, 0)
                    init_rot_quats.append(obj_init_rot_quat)
            options["model_ids"] = tuple(model_ids)
            obj_init_options["source_obj_id"] = 0
            obj_init_options["target_obj_id"] = 1
            obj_init_options["init_xys"] = init_xys
            obj_init_options["init_rot_quats"] = init_rot_quats
            options["obj_init_options"] = obj_init_options

            self._source_obj_name = options["model_ids"][obj_init_options["source_obj_id"]]
            self._target_obj_name = options["model_ids"][obj_init_options["target_obj_id"]]

            obs, info = super().reset(seed=self._episode_seed, options=options)
            return obs, info
        else:
            obj_init_options = options.get("obj_init_options", {})
            obj_init_options = obj_init_options.copy()
            episode_id = obj_init_options.get(
                "episode_id",
                self._episode_rng.randint(len(self._xy_configs) * len(self._quat_configs)),
            )
            xy_config = self._xy_configs[
                (episode_id % (len(self._xy_configs) * len(self._quat_configs)))
                // len(self._quat_configs)
                ]
            quat_config = self._quat_configs[episode_id % len(self._quat_configs)]

            options["model_ids"] = [self._source_obj_name, self._target_obj_name]
            obj_init_options["source_obj_id"] = 0
            obj_init_options["target_obj_id"] = 1
            obj_init_options["init_xys"] = xy_config
            obj_init_options["init_rot_quats"] = quat_config
            options["obj_init_options"] = obj_init_options

            obs, info = super().reset(seed=self._episode_seed, options=options)
            info.update({"episode_id": episode_id})
            return obs, info

    def _additional_prepackaged_config_reset(self, options):
        # use prepackaged robot evaluation configs under visual matching setup
        options["robot_init_options"] = {
            "init_xy": [0.147, 0.028],
            "init_rot_quat": [0, 0, 0, 1],
        }
        return False

    def _load_model(self):
        self.episode_objs = []
        for (model_id, model_scale) in zip(
                self.episode_model_ids, self.episode_model_scales
        ):
            density = self.model_db[model_id].get("density", 1000)

            obj = self._build_actor_helper(
                model_id,
                self._scene,
                scale=model_scale,
                density=density,
                physical_material=self._scene.create_physical_material(
                    static_friction=self.obj_static_friction,
                    dynamic_friction=self.obj_dynamic_friction,
                    restitution=0.0,
                ),
                root_dir=self.asset_root,
            )
            obj.name = model_id
            self.episode_objs.append(obj)


@register_env("PutOnCustomizable-v0", max_episode_steps=120)
class PutOnCustomizableInScene(PutOnBridgeInSceneEnv):
    def __init__(
            self,
            source_obj_name="bridge_spoon_generated_modified",
            target_obj_name="table_cloth_generated_shorter",
            **kwargs,
    ):
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = (
                grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None]
                + xy_center[None]
        )

        xy_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j:
                    xy_configs.append(np.array([grid_pos_1, grid_pos_2]))

        quat_configs = [
            np.array([[1, 0, 0, 0], [1, 0, 0, 0]]),
            np.array([euler2quat(0, 0, np.pi / 2), [1, 0, 0, 0]]),
        ]

        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def evaluate(self, success_require_src_completely_on_target=False, **kwargs):
        # this environment allows spoons to be partially on the table cloth to be considered successful
        return super().evaluate(success_require_src_completely_on_target, **kwargs)

    def get_language_instruction(self, **kwargs):
        src_name = self._get_instruction_obj_name(self._source_obj_name)
        tgt_name = self._get_instruction_obj_name(self._target_obj_name)
        return f"put the {src_name} on the {tgt_name}"


@register_env("PutOnCustomizableYCB-v0", max_episode_steps=120)
class PutOnBridgeInSceneYCBEnv(PutOnCustomizableInScene):
    DEFAULT_ASSET_ROOT = "{ASSET_DIR}/ycb-dataset"
    DEFAULT_MODEL_JSON = "info_ycb_put.json"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@register_env("PutOnCustomizableNoOverlay-v0", max_episode_steps=120)
class PutOnCustomizableNoOverlayInScene(PutOnCustomizableInScene):
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "widowx"
        ret["control_freq"] = 5
        ret["sim_freq"] = 500
        ret["control_mode"] = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
        ret["scene_name"] = "bridge_table_1_v1"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_cameras"] = ["3rd_view_camera"]

        return ret


@register_env("PutInCustomizable-v0", max_episode_steps=120)
class PutInCustomizableScene(PutOnBridgeInSceneEnv):
    def __init__(
            self,
            **kwargs,
    ):
        source_obj_name = "eggplant"
        target_obj_name = "dummy_sink_target_plane"  # invisible

        target_xy = np.array([-0.125, 0.025])
        xy_center = [-0.105, 0.206]

        half_span_x = 0.01
        half_span_y = 0.015
        num_x = 2
        num_y = 4

        grid_pos = []
        for x in np.linspace(-half_span_x, half_span_x, num_x):
            for y in np.linspace(-half_span_y, half_span_y, num_y):
                grid_pos.append(np.array([x + xy_center[0], y + xy_center[1]]))

        xy_configs = [np.stack([pos, target_xy], axis=0) for pos in grid_pos]

        quat_configs = [
            np.array([
                euler2quat(0, 0, 0, 'sxyz'),
                [1, 0, 0, 0]
            ]),
            np.array([
                euler2quat(0, 0, 1 * np.pi / 4, 'sxyz'),
                [1, 0, 0, 0]
            ]),
            np.array([
                euler2quat(0, 0, -1 * np.pi / 4, 'sxyz'),
                [1, 0, 0, 0]
            ]),
        ]

        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            rgb_always_overlay_objects=['sink', 'dummy_sink_target_plane'],
            **kwargs,
        )

    def get_language_instruction(self, **kwargs):
        src_name = self._get_instruction_obj_name(self._source_obj_name)
        tgt_name = self._get_instruction_obj_name(self._target_obj_name)
        return f"put the {src_name} into the yellow basket"

    def _load_model(self):
        super()._load_model()
        self.sink_id = 'sink'
        self.sink = self._build_actor_helper(
            self.sink_id,
            self._scene,
            density=self.model_db[self.sink_id].get("density", 1000),
            physical_material=self._scene.create_physical_material(
                static_friction=self.obj_static_friction, dynamic_friction=self.obj_dynamic_friction, restitution=0.0
            ),
            root_dir=self.asset_root,
        )
        self.sink.name = self.sink_id

    def _initialize_actors(self):
        # Move the robot far away to avoid collision
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))

        self.sink.set_pose(sapien.Pose(
            [-0.16, 0.13, 0.88],
            [1, 0, 0, 0]
        ))
        self.sink.lock_motion()

        super()._initialize_actors()

    def evaluate(self, *args, **kwargs):
        return super().evaluate(success_require_src_completely_on_target=False,
                                z_flag_required_offset=0.06,
                                *args, **kwargs)

    def _setup_prepackaged_env_init_config(self):
        ret = super()._setup_prepackaged_env_init_config()
        ret["robot"] = "widowx_sink_camera_setup"
        ret["scene_name"] = "bridge_table_1_v2"
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/bridge_sink.png"
        )
        return ret

    def _additional_prepackaged_config_reset(self, options):
        # use prepackaged robot evaluation configs under visual matching setup
        options["robot_init_options"] = {
            "init_xy": [0.127, 0.06],
            "init_rot_quat": [0, 0, 0, 1],
        }
        return False  # in env reset options, no need to reconfigure the environment


@register_env("PutInCustomizableYCB-v0", max_episode_steps=120)
class PutInBridgeInSceneYCBEnv(PutInCustomizableScene):
    DEFAULT_ASSET_ROOT = "{ASSET_DIR}/ycb-dataset"
    DEFAULT_MODEL_JSON = "info_ycb_put.json"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@register_env("PutInCustomizableNoOverlay-v0", max_episode_steps=120)
class PutInCustomizableNoOverlayScene(PutInCustomizableScene):
    def _setup_prepackaged_env_init_config(self):
        ret = super()._setup_prepackaged_env_init_config()
        ret["robot"] = "widowx_sink_camera_setup"
        ret["scene_name"] = "bridge_table_1_v2"
        return ret