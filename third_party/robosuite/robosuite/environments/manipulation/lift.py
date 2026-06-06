from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat


class Lift(ManipulationEnv):
    """
    This class corresponds to the lifting task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        base_types (None or str or list of str): type of base, used to instantiate base models from base factory.
            Default is "default", which is the default base associated with the robot(s) the 'robots' specification.
            None results in no base, and any other (valid) model overrides the default base. Should either be
            single str if same base type is to be used for all robots or else it should be a list of the same
            length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        lite_physics (bool): Whether to optimize for mujoco forward and step calls to reduce total simulation overhead.
            Set to False to preserve backward compatibility with datasets collected in robosuite <= 1.4.1.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.25 is provided if the cube is lifted

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube

        The sparse reward only consists of the lifting component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 2.25

        # use a shaping reward
        elif self.reward_shaping:

            # reaching reward
            dist = self._gripper_to_target(
                gripper=self.robots[0].gripper, target=self.cube.root_body, target_type="body", return_distance=True
            )
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward

            # grasping reward
            if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube):
                reward += 0.25

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cube = BoxObject(
            name="cube",
            size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
            size_max=[0.022, 0.022, 0.022],  # [0.018, 0.018, 0.018])
            rgba=[1, 0, 0, 1],
            material=redwood,
            rng=self.rng,
        )

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rng=self.rng,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # define observables modality
            modality = "object"

            # cube-related observables
            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube_body_id])

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")

            sensors = [cube_pos, cube_quat]

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])

            # gripper to cube position sensor; one for each arm
            sensors += [
                self._get_obj_eef_sensor(full_pf, "cube_pos", f"{arm_pf}gripper_to_cube_pos", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube)

    def _check_success(self):
        """
        Check if cube has been lifted.

        Returns:
            bool: True if cube has been lifted
        """
        cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        table_height = self.model.mujoco_arena.table_offset[2]
        success_height_margin = getattr(self, "success_height_margin", 0.04)

        # cube is higher than the table top above a margin
        return cube_height > table_height + success_height_margin


def _load_lift_variant_model(
    env,
    cube_size_min,
    cube_size_max,
    cube_rgba,
    cube_texture="WoodRed",
    cube_material_name="variant_cube_mat",
    placement_x_range=(-0.03, 0.03),
    placement_y_range=(-0.03, 0.03),
    table_material=None,
    agentview_camera=None,
):
    """
    Shared loader for explicit Lift robustness variants.

    This keeps the original Lift class unchanged while allowing evaluation-time
    object, color, and camera distribution shifts to be registered as separate
    robosuite environments.
    """
    super(Lift, env)._load_model()

    xpos = env.robots[0].robot_model.base_xpos_offset["table"](env.table_full_size[0])
    env.robots[0].robot_model.set_base_xpos(xpos)

    mujoco_arena = TableArena(
        table_full_size=env.table_full_size,
        table_friction=env.table_friction,
        table_offset=env.table_offset,
    )
    mujoco_arena.set_origin([0, 0, 0])

    if table_material is not None:
        mujoco_arena.table_visual.set("material", table_material)

    if agentview_camera is not None:
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=agentview_camera["pos"],
            quat=agentview_camera["quat"],
            camera_attribs=agentview_camera.get("attribs", None),
        )

    tex_attrib = {
        "type": "cube",
    }
    mat_attrib = {
        "texrepeat": "1 1",
        "specular": "0.4",
        "shininess": "0.1",
    }
    cube_material = CustomMaterial(
        texture=cube_texture,
        tex_name=f"{cube_material_name}_tex",
        mat_name=cube_material_name,
        tex_attrib=tex_attrib,
        mat_attrib=mat_attrib,
    )
    env.cube = BoxObject(
        name="cube",
        size_min=cube_size_min,
        size_max=cube_size_max,
        rgba=cube_rgba,
        material=cube_material,
        rng=env.rng,
    )

    if env.placement_initializer is not None:
        env.placement_initializer.reset()
        env.placement_initializer.add_objects(env.cube)
    else:
        env.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=env.cube,
            x_range=list(placement_x_range),
            y_range=list(placement_y_range),
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=env.table_offset,
            z_offset=0.01,
            rng=env.rng,
        )

    env.model = ManipulationTask(
        mujoco_arena=mujoco_arena,
        mujoco_robots=[robot.robot_model for robot in env.robots],
        mujoco_objects=env.cube,
    )


class LiftObjectPerturb(Lift):
    """
    Lift variant for object geometry OOD evaluation.

    The cube is larger and the initial placement range is wider than the clean
    Lift environment. Use this only for testing robustness, not as the clean
    baseline environment.
    """

    def _load_model(self):
        _load_lift_variant_model(
            env=self,
            cube_size_min=[0.026, 0.026, 0.026],
            cube_size_max=[0.030, 0.030, 0.030],
            cube_rgba=[1, 0, 0, 1],
            cube_texture="WoodRed",
            cube_material_name="large_redwood_mat",
            placement_x_range=(-0.055, 0.055),
            placement_y_range=(-0.055, 0.055),
        )


class LiftColorPerturb(Lift):
    """
    Lift variant for color / texture OOD evaluation.

    The cube uses a blue wood texture and the table switches to a flat table
    material. This avoids DomainRandomizationWrapper color randomization, which
    is brittle across MuJoCo versions in this codebase.
    """

    def _load_model(self):
        _load_lift_variant_model(
            env=self,
            cube_size_min=[0.020, 0.020, 0.020],
            cube_size_max=[0.022, 0.022, 0.022],
            cube_rgba=[0.05, 0.25, 1.0, 1],
            cube_texture="WoodBlue",
            cube_material_name="bluewood_mat",
            table_material="table_mat",
        )


class LiftCameraPerturb(Lift):
    """
    Lift variant for camera OOD evaluation.

    The external agentview camera is shifted and given a wider field of view.
    The wrist camera remains unchanged, so failures can be attributed to a
    realistic external-view shift instead of a completely different sensor rig.
    """

    def _load_model(self):
        _load_lift_variant_model(
            env=self,
            cube_size_min=[0.020, 0.020, 0.020],
            cube_size_max=[0.022, 0.022, 0.022],
            cube_rgba=[1, 0, 0, 1],
            cube_texture="WoodRed",
            cube_material_name="camera_redwood_mat",
            agentview_camera={
                "pos": np.array([0.62, -0.10, 1.42]),
                "quat": np.array([0.653, 0.271, 0.271, 0.653]),
                "attribs": {"fovy": "70"},
            },
        )


class LiftVisualOOD(Lift):
    """
    Lift variant for a clean visual-only OOD shift.

    Geometry, placement range, friction, and success threshold are kept the
    same as the clean Lift task. Only appearance and the external camera are
    shifted so this environment isolates visual robustness.
    """

    def _load_model(self):
        _load_lift_variant_model(
            env=self,
            cube_size_min=[0.020, 0.020, 0.020],
            cube_size_max=[0.022, 0.022, 0.022],
            cube_rgba=[0.10, 0.22, 0.95, 1.0],
            cube_texture="WoodBlue",
            cube_material_name="visual_ood_bluewood_mat",
            table_material="table_mat",
            agentview_camera={
                "pos": np.array([0.60, -0.12, 1.40]),
                "quat": np.array([0.671, 0.224, 0.224, 0.671]),
                "attribs": {"fovy": "68"},
            },
        )


class LiftTaskDynamicsHard(Lift):
    """
    Lift variant that makes the task and contact dynamics harder.

    Visual appearance stays close to the clean Lift setup, while the object is
    smaller, the placement range is wider, the table is more slippery, and the
    success threshold is higher.
    """

    def __init__(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["table_friction"] = (0.35, 2e-3, 5e-5)
        if kwargs.get("initialization_noise", "default") == "default":
            kwargs["initialization_noise"] = {"magnitude": 0.01, "type": "uniform"}
        self.success_height_margin = 0.055
        super().__init__(*args, **kwargs)

    def _load_model(self):
        _load_lift_variant_model(
            env=self,
            cube_size_min=[0.017, 0.017, 0.017],
            cube_size_max=[0.019, 0.019, 0.019],
            cube_rgba=[1, 0, 0, 1],
            cube_texture="WoodRed",
            cube_material_name="task_dynamics_redwood_mat",
            placement_x_range=(-0.065, 0.065),
            placement_y_range=(-0.065, 0.065),
        )
