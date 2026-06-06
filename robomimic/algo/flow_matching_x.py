"""
Flow Matching Policy with x-prediction.

This variant uses the same straight-line interpolation path as flow_matching.py,
and parameterizes the velocity field through a clean action trajectory x_0
prediction.
"""
import torch
import torch.nn.functional as F

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.algo.flow_matching import FlowMatchingPolicyUNet


@register_algo_factory_func("flow_matching_x")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the Flow Matching x-prediction algo class.
    """
    if algo_config.unet.enabled:
        return FlowMatchingXPolicyUNet, {}
    raise RuntimeError("Flow Matching x-prediction requires algo.unet.enabled = True")


class FlowMatchingXPolicyUNet(FlowMatchingPolicyUNet):
    """
    Flow Matching where the UNet predicts clean actions x_0 and the loss is
    applied to the implied path velocity.

    Training:
        x_t = (1 - t) * noise + t * actions
        x_pred = model(x_t, t, obs_cond)
        v_pred = (x_pred - x_t) / (1 - t)
        loss = MSE(v_pred, actions - noise)

    Inference:
        Convert x-prediction into the corresponding velocity field:
        v = (x_pred - x_t) / (1 - t)
    """

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.
        """
        B = batch["actions"].shape[0]

        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = PolicyAlgo.train_on_batch(self, batch, epoch, validate=validate)
            actions = batch["actions"]

            inputs = {
                "obs": batch["obs"],
                "goal": batch["goal_obs"],
            }
            for k in self.obs_shapes:
                assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])

            obs_features = TensorUtils.time_distributed(
                inputs,
                self.nets["policy"]["obs_encoder"],
                inputs_as_kwargs=True,
            )
            assert obs_features.ndim == 3
            obs_cond = obs_features.flatten(start_dim=1)

            noise = torch.randn(actions.shape, device=self.device)

            num_bins = self.algo_config.flow.num_train_timesteps
            t_bins = torch.randint(0, num_bins, (B,), device=self.device)
            t = t_bins.float() / num_bins
            t_expand = t[:, None, None]

            x_t = (1.0 - t_expand) * noise + t_expand * actions

            v_target = actions - noise

            x_pred = self.nets["policy"]["noise_pred_net"](
                x_t,
                t,
                global_cond=obs_cond,
            )

            v_pred = self._x_pred_to_velocity(
                x_t=x_t,
                x_pred=x_pred,
                t=t,
                min_denom=1.0 / num_bins,
            )
            loss = F.mse_loss(v_pred, v_target)

            losses = {"v_loss": loss}
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                )

                if self.ema is not None:
                    self.ema.step(self.nets)

                info.update({"policy_grad_norms": policy_grad_norms})

        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch for logging.
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["v_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    @staticmethod
    def _x_pred_to_velocity(x_t, x_pred, t, min_denom=1e-4):
        """
        Convert clean-sample prediction to rectified-flow velocity.
        """
        denom = (1.0 - t[:, None, None]).clamp_min(min_denom)
        return (x_pred - x_t) / denom

    def _get_action_trajectory(self, obs_dict, goal_dict=None):
        """
        Run the ODE solver using x_0 predictions converted to velocities.
        """
        assert not self.nets.training
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        num_steps = self.algo_config.flow.num_inference_steps
        solver = self.algo_config.flow.solver

        nets = self.nets
        if self.ema is not None:
            nets = self.ema.averaged_model

        inputs = {
            "obs": obs_dict,
            "goal": goal_dict,
        }
        for k in self.obs_shapes:
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])

        obs_features = TensorUtils.time_distributed(
            inputs,
            nets["policy"]["obs_encoder"],
            inputs_as_kwargs=True,
        )
        assert obs_features.ndim == 3
        B = obs_features.shape[0]
        obs_cond = obs_features.flatten(start_dim=1)

        x_t = torch.randn((B, Tp, action_dim), device=self.device)
        dt = 1.0 / num_steps

        if solver == "euler":
            for i in range(num_steps):
                t = torch.full((B,), i / num_steps, device=self.device)
                x_pred = nets["policy"]["noise_pred_net"](
                    sample=x_t,
                    timestep=t,
                    global_cond=obs_cond,
                )
                v_pred = self._x_pred_to_velocity(x_t=x_t, x_pred=x_pred, t=t, min_denom=dt)
                x_t = x_t + v_pred * dt

        elif solver == "heun":
            for i in range(num_steps):
                t_val = i / num_steps
                t = torch.full((B,), t_val, device=self.device)

                x_pred_1 = nets["policy"]["noise_pred_net"](
                    sample=x_t,
                    timestep=t,
                    global_cond=obs_cond,
                )
                v1 = self._x_pred_to_velocity(x_t=x_t, x_pred=x_pred_1, t=t, min_denom=dt)
                x_euler = x_t + v1 * dt

                t_next_val = (i + 1) / num_steps
                t_next = torch.full((B,), t_next_val, device=self.device)
                x_pred_2 = nets["policy"]["noise_pred_net"](
                    sample=x_euler,
                    timestep=t_next,
                    global_cond=obs_cond,
                )
                v2 = self._x_pred_to_velocity(x_t=x_euler, x_pred=x_pred_2, t=t_next, min_denom=dt)

                x_t = x_t + 0.5 * (v1 + v2) * dt
        else:
            raise ValueError("Unknown ODE solver: {}. Use 'euler' or 'heun'.".format(solver))

        start = To - 1
        end = start + Ta
        return x_t[:, start:end]
