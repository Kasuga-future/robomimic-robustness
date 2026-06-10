"""
Staged-reward-guided reward-weighted flow finetuning.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.algo.flow_matching_x import FlowMatchingXPolicyUNet


@register_algo_factory_func("flow_rwr")
def algo_config_to_class(algo_config):
    if algo_config.unet.enabled:
        return FlowRWRPolicyUNet, {}
    raise RuntimeError("Flow RWR requires algo.unet.enabled = True")


class FlowRWRPolicyUNet(FlowMatchingXPolicyUNet):
    """
    Reward-weighted online finetuning on top of the x-prediction flow policy.
    """

    def sample_action_chunk(self, obs_dict, goal_dict=None):
        self.set_eval()
        with torch.no_grad():
            action_chunk = self._get_action_trajectory(obs_dict=obs_dict, goal_dict=goal_dict)
        return action_chunk

    def process_online_batch_for_training(self, batch):
        batch = TensorUtils.to_tensor(batch)
        batch = TensorUtils.to_device(batch, self.device)
        batch = TensorUtils.to_float(batch)
        return batch

    def _compute_obs_cond(self, obs_batch, goal_batch=None):
        inputs = {
            "obs": obs_batch,
            "goal": goal_batch,
        }
        for key in self.obs_shapes:
            assert inputs["obs"][key].ndim - 2 == len(self.obs_shapes[key])
        obs_features = TensorUtils.time_distributed(
            inputs,
            self.nets["policy"]["obs_encoder"],
            inputs_as_kwargs=True,
        )
        assert obs_features.ndim == 3
        return obs_features.flatten(start_dim=1)

    def _compute_flow_losses(self, obs_batch, actions, goal_batch=None, action_mask=None):
        batch_size = actions.shape[0]
        obs_cond = self._compute_obs_cond(obs_batch=obs_batch, goal_batch=goal_batch)

        noise = torch.randn(actions.shape, device=self.device)
        num_bins = self.algo_config.flow.num_train_timesteps
        t_bins = torch.randint(0, num_bins, (batch_size,), device=self.device)
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

        loss_per_elem = F.mse_loss(v_pred, v_target, reduction="none").mean(dim=-1)
        if action_mask is not None:
            mask = action_mask.float()
            denom = mask.sum(dim=1).clamp_min(1.0)
            loss_per_sample = (loss_per_elem * mask).sum(dim=1) / denom
        else:
            loss_per_sample = loss_per_elem.mean(dim=1)
        return loss_per_sample

    def compute_demo_loss(self, batch):
        loss_per_sample = self._compute_flow_losses(
            obs_batch=batch["obs"],
            actions=batch["actions"],
            goal_batch=batch.get("goal_obs", None),
            action_mask=None,
        )
        return loss_per_sample.mean(), loss_per_sample

    def compute_weighted_online_loss(self, rollout_batch):
        loss_per_sample = self._compute_flow_losses(
            obs_batch=rollout_batch["obs"],
            actions=rollout_batch["actions"],
            goal_batch=rollout_batch.get("goal_obs", None),
            action_mask=rollout_batch.get("action_mask", None),
        )
        weights = rollout_batch["weights"].float().clamp_min(0.0)
        weighted_loss = (weights * loss_per_sample).mean()
        return weighted_loss, loss_per_sample

    def train_on_mixed_batch(self, demo_batch, online_batch, epoch, validate=False):
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = PolicyAlgo.train_on_batch(self, demo_batch if demo_batch is not None else online_batch, epoch, validate=validate)

            total_loss = torch.tensor(0.0, device=self.device)
            losses = OrderedDict()

            if demo_batch is not None:
                demo_loss, demo_loss_per_sample = self.compute_demo_loss(demo_batch)
                total_loss = total_loss + demo_loss
                losses["demo_loss"] = demo_loss
                info["demo_loss_per_sample"] = TensorUtils.detach(demo_loss_per_sample)
            else:
                losses["demo_loss"] = torch.tensor(0.0, device=self.device)

            if online_batch is not None:
                online_loss, online_loss_per_sample = self.compute_weighted_online_loss(online_batch)
                lambda_online = float(self.algo_config.rwr.lambda_online)
                total_loss = total_loss + lambda_online * online_loss
                losses["online_loss"] = online_loss
                info["online_loss_per_sample"] = TensorUtils.detach(online_loss_per_sample)
                info["weights"] = TensorUtils.detach(online_batch["weights"])
                info["returns"] = TensorUtils.detach(online_batch["returns"])
                info["advantages"] = TensorUtils.detach(online_batch["advantages"])
            else:
                losses["online_loss"] = torch.tensor(0.0, device=self.device)

            losses["total_loss"] = total_loss
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=total_loss,
                    max_grad_norm=self.global_config.train.max_grad_norm,
                )
                if self.ema is not None:
                    self.ema.step(self.nets)
                info["policy_grad_norms"] = policy_grad_norms

        return info

    def train_on_batch(self, batch, epoch, validate=False):
        return self.train_on_mixed_batch(demo_batch=batch, online_batch=None, epoch=epoch, validate=validate)

    def log_info(self, info):
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["total_loss"].item()
        log["Demo_Loss"] = info["losses"]["demo_loss"].item()
        log["Online_Loss"] = info["losses"]["online_loss"].item()
        if "weights" in info:
            log["Online_Weight_Mean"] = info["weights"].mean().item()
        if "returns" in info:
            log["Online_Return_Mean"] = info["returns"].mean().item()
        if "advantages" in info:
            log["Online_Adv_Mean"] = info["advantages"].mean().item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

