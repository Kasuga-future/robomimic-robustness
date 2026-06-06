"""
Config for Flow Matching x-prediction policy.
"""

from robomimic.config.flow_matching_config import FlowMatchingConfig


class FlowMatchingXConfig(FlowMatchingConfig):
    ALGO_NAME = "flow_matching_x"

    def train_config(self):
        super(FlowMatchingXConfig, self).train_config()
        self.train.output_dir = "./flow_matching_x_image_eval_logs"
