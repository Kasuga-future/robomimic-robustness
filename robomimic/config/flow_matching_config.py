"""
Config for Flow Matching Policy algorithm.

Flow Matching replaces the DDPM/DDIM diffusion process with a straight-line
interpolation path (rectified flow) and an ODE solver for inference.
This removes the dependency on HuggingFace diffusers schedulers.
"""

from robomimic.config.base_config import BaseConfig


class FlowMatchingConfig(BaseConfig):
    ALGO_NAME = "flow_matching"

    def train_config(self):
        """
        Setting up training parameters for Flow Matching Policy.

        - don't need "next_obs" from hdf5 - so save on storage and compute by disabling it
        - set compatible data loading parameters
        """
        super(FlowMatchingConfig, self).train_config()

        # disable next_obs loading from hdf5
        self.train.hdf5_load_next_obs = False

        # keep a distinct default output root for image-based flow matching experiments
        self.train.output_dir = "./flow_matching_image_eval_logs"

        # set compatible data loading parameters
        self.train.seq_length = 16   # should match self.algo.horizon.prediction_horizon
        self.train.frame_stack = 2   # should match self.algo.horizon.observation_horizon

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config`
        argument to the constructor. Any parameter that an algorithm needs to determine its
        training and test-time behavior should be populated here.
        """

        # optimization parameters
        self.algo.optim_params.policy.optimizer_type = "adamw"
        self.algo.optim_params.policy.learning_rate.initial = 1e-4      # policy learning rate
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.policy.learning_rate.step_every_batch = True
        self.algo.optim_params.policy.learning_rate.scheduler_type = "cosine"
        self.algo.optim_params.policy.learning_rate.num_cycles = 0.5    # number of cosine cycles
        self.algo.optim_params.policy.learning_rate.warmup_steps = 500  # number of warmup steps
        self.algo.optim_params.policy.learning_rate.epoch_schedule = []
        self.algo.optim_params.policy.learning_rate.do_not_lock_keys()
        self.algo.optim_params.policy.regularization.L2 = 1e-6          # L2 regularization strength

        # horizon parameters
        self.algo.horizon.observation_horizon = 2
        self.algo.horizon.action_horizon = 8
        self.algo.horizon.prediction_horizon = 16

        # UNet parameters (same architecture as diffusion policy)
        self.algo.unet.enabled = True
        self.algo.unet.diffusion_step_embed_dim = 256
        self.algo.unet.down_dims = [256, 512, 1024]
        self.algo.unet.kernel_size = 5
        self.algo.unet.n_groups = 8

        # EMA parameters
        self.algo.ema.enabled = True
        self.algo.ema.power = 0.75

        # Flow Matching parameters
        self.algo.flow.num_train_timesteps = 100    # number of discrete bins for sampling t during training
        self.algo.flow.num_inference_steps = 100    # align default sampling steps with diffusion-policy configs
        self.algo.flow.solver = "euler"             # ODE solver type: "euler" or "heun"

    def observation_config(self):
        """
        Default to the same proprio + RGB observation setup used by the image diffusion configs.
        """
        super(FlowMatchingConfig, self).observation_config()

        self.observation.modalities.obs.low_dim = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "robot0_joint_pos",
            "robot0_joint_vel",
        ]
        self.observation.modalities.obs.rgb = [
            "agentview_image",
            "robot0_eye_in_hand_image",
        ]
        self.observation.modalities.obs.depth = []
        self.observation.modalities.obs.scan = []
        self.observation.modalities.goal.low_dim = []
        self.observation.modalities.goal.rgb = []
        self.observation.modalities.goal.depth = []
        self.observation.modalities.goal.scan = []

        self.observation.encoder.rgb.core_class = "VisualCore"
        self.observation.encoder.rgb.core_kwargs.feature_dimension = 64
        self.observation.encoder.rgb.core_kwargs.backbone_class = "ResNet18Conv"
        self.observation.encoder.rgb.core_kwargs.backbone_kwargs.pretrained = False
        self.observation.encoder.rgb.core_kwargs.backbone_kwargs.input_coord_conv = False
        self.observation.encoder.rgb.core_kwargs.pool_class = "SpatialSoftmax"
        self.observation.encoder.rgb.core_kwargs.pool_kwargs.num_kp = 32
        self.observation.encoder.rgb.core_kwargs.pool_kwargs.learnable_temperature = False
        self.observation.encoder.rgb.core_kwargs.pool_kwargs.temperature = 1.0
        self.observation.encoder.rgb.core_kwargs.pool_kwargs.noise_std = 0.0
