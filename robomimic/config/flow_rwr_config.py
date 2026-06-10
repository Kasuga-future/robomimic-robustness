"""
Config for staged-reward-guided reward-weighted flow finetuning.
"""

from robomimic.config.flow_matching_x_config import FlowMatchingXConfig


class FlowRWRConfig(FlowMatchingXConfig):
    ALGO_NAME = "flow_rwr"

    def experiment_config(self):
        super(FlowRWRConfig, self).experiment_config()
        self.experiment.rollout.enabled = False
        self.experiment.render_video = True
        self.experiment.keep_all_videos = True

    def train_config(self):
        super(FlowRWRConfig, self).train_config()
        self.train.output_dir = "./flow_rwr_finetune_logs"

        self.train.online.checkpoint_path = None
        self.train.online.env_name = None
        self.train.online.num_iters = 100
        self.train.online.num_rollout_episodes_per_iter = 10
        self.train.online.num_train_steps_per_iter = 100
        self.train.online.rollout_horizon = 400
        self.train.online.terminate_on_success = True
        self.train.online.demo_batch_ratio = 0.5
        self.train.online.eval_interval = 10
        self.train.online.save_interval = 10
        self.train.online.num_eval_episodes = 10

    def algo_config(self):
        super(FlowRWRConfig, self).algo_config()
        self.algo.rwr.gamma = 0.99
        self.algo.rwr.reward_temperature = 1.0
        self.algo.rwr.min_weight = 0.0
        self.algo.rwr.max_weight = 5.0
        self.algo.rwr.topk_fraction = 1.0
        self.algo.rwr.lambda_online = 1.0
        self.algo.rwr.use_staged_reward = True
        self.algo.rwr.use_segment_level_weighting = True
        self.algo.rwr.success_bonus = 5.0
        self.algo.rwr.advantage_eps = 1e-6

