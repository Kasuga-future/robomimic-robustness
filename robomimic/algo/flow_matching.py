"""
Implementation of Flow Matching Policy.

Flow Matching (a.k.a. Rectified Flow) replaces the DDPM/DDIM diffusion process
with a straight-line interpolation path between noise and data, and uses an ODE
solver for inference. This removes the dependency on HuggingFace diffusers.

Reference: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023.
           Esser et al., "Scaling Rectified Flow Transformers", ICML 2024.
"""
from typing import Callable
from collections import OrderedDict, deque
from packaging.version import parse as parse_version
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.models.obs_nets as ObsNets
import robomimic.models.diffusion_policy_nets as DPNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo


@register_algo_factory_func("flow_matching")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the Flow Matching algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    if algo_config.unet.enabled:
        return FlowMatchingPolicyUNet, {}
    else:
        raise RuntimeError("Flow Matching requires algo.unet.enabled = True")


class SimpleEMA:
    """
    A lightweight Exponential Moving Average of model parameters.
    Replaces the HuggingFace diffusers EMAModel dependency.
    """

    def __init__(self, model: nn.Module, power: float = 0.75):
        self.power = power
        self.averaged_model = copy.deepcopy(model)
        self.averaged_model.requires_grad_(False)
        self._num_updates = 0

    def step(self, model: nn.Module):
        """Update EMA weights from the current model."""
        self._num_updates += 1
        decay = 1.0 - (1.0 + self._num_updates) ** (-self.power)
        with torch.no_grad():
            for ema_p, p in zip(self.averaged_model.parameters(), model.parameters()):
                ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


class FlowMatchingPolicyUNet(PolicyAlgo):

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)

        obs_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        obs_encoder = replace_bn_with_gn(obs_encoder)

        obs_dim = obs_encoder.output_shape()[0]

        # create network object -- same ConditionalUnet1D as diffusion policy
        velocity_pred_net = DPNets.ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=obs_dim * self.algo_config.horizon.observation_horizon,
            diffusion_step_embed_dim=self.algo_config.unet.diffusion_step_embed_dim,
            down_dims=self.algo_config.unet.down_dims,
            kernel_size=self.algo_config.unet.kernel_size,
            n_groups=self.algo_config.unet.n_groups,
        )

        # the final arch has 2 parts
        nets = nn.ModuleDict({
            "policy": nn.ModuleDict({
                "obs_encoder": obs_encoder,
                "noise_pred_net": velocity_pred_net,  # reuse same key name for compat
            })
        })

        nets = nets.float().to(self.device)

        # setup EMA (self-contained, no diffusers dependency)
        ema = None
        if self.algo_config.ema.enabled:
            ema = SimpleEMA(model=nets, power=self.algo_config.ema.power)

        # set attrs
        self.nets = nets
        self.ema = ema
        self.action_check_done = False
        self.action_queue = None

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        To = self.algo_config.horizon.observation_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, :To, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None)
        input_batch["actions"] = batch["actions"][:, :Tp, :]

        # check if actions are normalized to [-1,1]
        if not self.action_check_done:
            actions = input_batch["actions"]
            in_range = (-1 <= actions) & (actions <= 1)
            all_in_range = torch.all(in_range).item()
            if not all_in_range:
                raise ValueError(
                    "'actions' must be in range [-1,1] for Flow Matching! "
                    "Check the dataset action range and train.action_config normalization."
                )
            self.action_check_done = True

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Flow Matching training:
            1. Sample t ~ Uniform(0, 1)
            2. Interpolate: x_t = (1 - t) * noise + t * actions
            3. Target velocity: v_target = actions - noise
            4. Predict velocity: v_pred = model(x_t, t, obs_cond)
            5. Loss = MSE(v_pred, v_target)

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
            epoch (int): epoch number
            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
        """
        B = batch["actions"].shape[0]

        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(FlowMatchingPolicyUNet, self).train_on_batch(batch, epoch, validate=validate)
            actions = batch["actions"]

            # encode obs
            inputs = {
                "obs": batch["obs"],
                "goal": batch["goal_obs"],
            }
            for k in self.obs_shapes:
                assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])

            obs_features = TensorUtils.time_distributed(inputs, self.nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
            assert obs_features.ndim == 3  # [B, T, D]

            obs_cond = obs_features.flatten(start_dim=1)

            # sample noise
            noise = torch.randn(actions.shape, device=self.device)

            # sample time t from discrete bins in [0, 1)
            # num_train_timesteps controls the granularity of binning
            num_bins = self.algo_config.flow.num_train_timesteps
            t_bins = torch.randint(0, num_bins, (B,), device=self.device)
            t = t_bins.float() / num_bins  # map to [0, 1)
            t_expand = t[:, None, None]  # (B, 1, 1) for broadcasting to (B, T, Da)

            # linear interpolation: x_t = (1 - t) * noise + t * x_0
            x_t = (1.0 - t_expand) * noise + t_expand * actions

            # velocity target: v = x_0 - noise (straight-line direction)
            v_target = actions - noise

            # predict velocity (reuse the same UNet, timestep is float in [0,1])
            v_pred = self.nets["policy"]["noise_pred_net"](
                x_t, t, global_cond=obs_cond
            )

            # L2 loss on velocity prediction
            loss = F.mse_loss(v_pred, v_target)

            # logging
            losses = {"l2_loss": loss}
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                # gradient step
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                )

                # update Exponential Moving Average of the model weights
                if self.ema is not None:
                    self.ema.step(self.nets)

                step_info = {
                    "policy_grad_norms": policy_grad_norms,
                }
                info.update(step_info)

        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(FlowMatchingPolicyUNet, self).log_info(info)
        log["Loss"] = info["losses"]["l2_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.

        Note: observation history stacking is handled externally by the
        environment's FrameStackWrapper, not by this class.
        """
        Ta = self.algo_config.horizon.action_horizon
        self.action_queue = deque(maxlen=Ta)

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation [1, Do]
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor [1, Da]
        """
        Ta = self.algo_config.horizon.action_horizon

        if len(self.action_queue) == 0:
            # no actions left, run inference
            # [1,T,Da]
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict)

            # put actions into the queue
            self.action_queue.extend(action_sequence[0])

        # has action, execute from left to right
        action = self.action_queue.popleft()
        action = action.unsqueeze(0)
        return action

    def _get_action_trajectory(self, obs_dict, goal_dict=None):
        """
        Core inference method. Runs the ODE solver from noise to clean actions.

        Supports two solvers:
            - "euler":   x_{t+dt} = x_t + v(x_t, t) * dt
            - "heun":    x_{t+dt} = x_t + 0.5 * (v1 + v2) * dt  (2nd-order)
        """
        assert not self.nets.training
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        num_steps = self.algo_config.flow.num_inference_steps
        solver = self.algo_config.flow.solver

        # select network (EMA averaged model for better inference)
        nets = self.nets
        if self.ema is not None:
            nets = self.ema.averaged_model

        # encode obs
        inputs = {
            "obs": obs_dict,
            "goal": goal_dict,
        }
        for k in self.obs_shapes:
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                # adding time dimension if not present
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
        obs_features = TensorUtils.time_distributed(inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
        assert obs_features.ndim == 3  # [B, T, D]
        B = obs_features.shape[0]

        # reshape observation to (B, obs_horizon * obs_dim)
        obs_cond = obs_features.flatten(start_dim=1)

        # initialize from pure Gaussian noise
        x_t = torch.randn((B, Tp, action_dim), device=self.device)

        dt = 1.0 / num_steps

        if solver == "euler":
            # Euler method: 1st order ODE solver
            for i in range(num_steps):
                t = torch.full((B,), i / num_steps, device=self.device)
                v_pred = nets["policy"]["noise_pred_net"](
                    sample=x_t, timestep=t, global_cond=obs_cond
                )
                x_t = x_t + v_pred * dt

        elif solver == "heun":
            # Heun's method: 2nd order ODE solver (predictor-corrector)
            for i in range(num_steps):
                t_val = i / num_steps
                t = torch.full((B,), t_val, device=self.device)

                # predictor: Euler step
                v1 = nets["policy"]["noise_pred_net"](
                    sample=x_t, timestep=t, global_cond=obs_cond
                )
                x_euler = x_t + v1 * dt

                # corrector: evaluate at the predicted point
                t_next_val = (i + 1) / num_steps
                t_next = torch.full((B,), t_next_val, device=self.device)
                v2 = nets["policy"]["noise_pred_net"](
                    sample=x_euler, timestep=t_next, global_cond=obs_cond
                )

                # average the two velocities
                x_t = x_t + 0.5 * (v1 + v2) * dt
        else:
            raise ValueError(f"Unknown ODE solver: {solver}. Use 'euler' or 'heun'.")

        # extract the action horizon window
        start = To - 1
        end = start + Ta
        action = x_t[:, start:end]
        return action

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return {
            "nets": self.nets.state_dict(),
            "optimizers": {k: self.optimizers[k].state_dict() for k in self.optimizers},
            "lr_schedulers": {k: self.lr_schedulers[k].state_dict() if self.lr_schedulers[k] is not None else None for k in self.lr_schedulers},
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict, load_optimizers=False):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
            load_optimizers (bool): whether to load optimizers and lr_schedulers from the model_dict;
                used when resuming training from a checkpoint
        """
        self.nets.load_state_dict(model_dict["nets"])

        # for backwards compatibility
        if "optimizers" not in model_dict:
            model_dict["optimizers"] = {}
        if "lr_schedulers" not in model_dict:
            model_dict["lr_schedulers"] = {}

        if model_dict.get("ema", None) is not None:
            self.ema.averaged_model.load_state_dict(model_dict["ema"])

        if load_optimizers:
            for k in model_dict["optimizers"]:
                self.optimizers[k].load_state_dict(model_dict["optimizers"][k])
            for k in model_dict["lr_schedulers"]:
                if model_dict["lr_schedulers"][k] is not None:
                    self.lr_schedulers[k].load_state_dict(model_dict["lr_schedulers"][k])


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    if parse_version(torch.__version__) < parse_version("1.9.0"):
        raise ImportError("This function requires pytorch >= 1.9.0")

    bn_list = [k.split(".") for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split(".") for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int = 16) -> nn.Module:
    """
    Replace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features // features_per_group,
            num_channels=x.num_features)
    )
    return root_module
