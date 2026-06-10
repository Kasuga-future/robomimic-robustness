"""
Online staged-reward-guided reward-weighted flow finetuning.
"""
import argparse
import json
import os
import shutil
import sys
import traceback
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.python_utils as PyUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.algo import RolloutPolicy, algo_factory
from robomimic.config import config_factory
from robomimic.envs.reward_wrappers import StagedRewardWrapper
from robomimic.utils.flow_rollout_buffer import FlowRolloutBuffer
from robomimic.utils.log_utils import DataLogger, PrintLogger, flush_warnings


def tensor_action_chunk_to_numpy(rollout_policy, action_chunk):
    if torch.is_tensor(action_chunk):
        action_chunk = action_chunk.detach().cpu().numpy()

    if rollout_policy.action_normalization_stats is None:
        return action_chunk

    action_keys = rollout_policy.policy.global_config.train.action_keys
    action_shapes = {
        key: rollout_policy.action_normalization_stats[key]["offset"].shape[1:]
        for key in rollout_policy.action_normalization_stats
    }
    action_dict = PyUtils.vector_to_action_dict(
        action_chunk,
        action_shapes=action_shapes,
        action_keys=action_keys,
    )
    action_dict = ObsUtils.unnormalize_dict(
        action_dict,
        normalization_stats=rollout_policy.action_normalization_stats,
    )
    action_config = rollout_policy.policy.global_config.train.action_config
    for key, value in action_dict.items():
        this_format = action_config[key].get("format", None)
        if this_format == "rot_6d":
            rot_6d = torch.from_numpy(value)
            conversion_format = action_config[key].get("convert_at_runtime", "rot_axis_angle")
            if conversion_format == "rot_axis_angle":
                rot = TorchUtils.rot_6d_to_axis_angle(rot_6d=rot_6d).numpy()
            elif conversion_format == "rot_euler":
                rot = TorchUtils.rot_6d_to_euler_angles(rot_6d=rot_6d, convention="XYZ").numpy()
            else:
                raise ValueError("unknown rotation conversion format: {}".format(conversion_format))
            action_dict[key] = rot
    return PyUtils.action_dict_to_vector(action_dict, action_keys=action_keys)


def sample_action_chunk(rollout_policy, model, obs):
    obs_t = rollout_policy._prepare_observation(obs, batched_ob=False)
    with torch.no_grad():
        action_chunk = model.sample_action_chunk(obs_dict=obs_t)
    action_chunk = action_chunk[0]
    return tensor_action_chunk_to_numpy(rollout_policy, action_chunk)


def maybe_wrap_env_for_online_training(env, config):
    env = EnvUtils.wrap_env_from_config(env, config=config)
    env = StagedRewardWrapper(
        env,
        use_staged_reward=config.algo.rwr.use_staged_reward,
        success_bonus=config.algo.rwr.success_bonus,
    )
    return env


def create_envs(config, env_meta, shape_meta):
    env_meta = deepcopy(env_meta)
    env_meta["lang"] = None

    env_name = config.train.online.env_name if config.train.online.env_name is not None else env_meta["env_name"]
    env_names = [env_name]
    if config.experiment.additional_envs is not None:
        env_names.extend(config.experiment.additional_envs)

    envs = OrderedDict()
    for name in env_names:
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            env_name=name,
            render=False,
            render_offscreen=(config.experiment.render_video or shape_meta["use_images"] or shape_meta["use_depths"]),
            use_image_obs=shape_meta["use_images"] or shape_meta["use_depths"],
            use_depth_obs=shape_meta["use_depths"],
        )
        env = maybe_wrap_env_for_online_training(env, config)
        envs[name] = env
    return envs


def build_model(config, shape_meta, device, ckpt_dict):
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    model.deserialize(ckpt_dict["model"])
    return model


def collect_rollout_episode(env, rollout_policy, model, episode_id, horizon, terminate_on_success):
    action_horizon = model.algo_config.horizon.action_horizon
    obs = env.reset()
    model.reset()

    steps = []
    segments = []
    timestep = 0
    success = False

    while timestep < horizon:
        chunk_obs = deepcopy(obs)
        action_chunk = sample_action_chunk(rollout_policy, model, chunk_obs)

        executed_actions = []
        rewards = []
        staged_infos = []
        done = False

        for action in action_chunk:
            next_obs, reward, env_done, info = env.step(action)
            step_success = bool(info.get("success", False))
            done = bool(env_done or (terminate_on_success and step_success) or ((timestep + 1) >= horizon))
            staged_info = {
                "r_reach": float(info.get("r_reach", 0.0)),
                "r_grasp": float(info.get("r_grasp", 0.0)),
                "r_lift": float(info.get("r_lift", 0.0)),
                "r_hover": float(info.get("r_hover", 0.0)),
                "r_success": float(info.get("r_success", 0.0)),
                "success": float(step_success),
                "reward_progress": float(info.get("reward_progress", reward)),
                "reward_total": float(info.get("reward_total", reward)),
                "base_reward": float(info.get("base_reward", reward)),
            }
            steps.append({
                "reward": float(reward),
                "done": bool(done),
                "success": bool(step_success),
                "staged": staged_info,
            })
            executed_actions.append(np.array(action, copy=True))
            rewards.append(float(reward))
            staged_infos.append(staged_info)
            obs = next_obs
            timestep += 1
            success = success or step_success
            if done:
                break

        if len(executed_actions) == 0:
            break

        action_dim = executed_actions[0].shape[-1]
        padded_actions = np.zeros((action_horizon, action_dim), dtype=np.float32)
        action_mask = np.zeros((action_horizon,), dtype=np.float32)
        exec_arr = np.stack(executed_actions, axis=0).astype(np.float32)
        padded_actions[:exec_arr.shape[0]] = exec_arr
        action_mask[:exec_arr.shape[0]] = 1.0

        staged_summary = {}
        if len(staged_infos) > 0:
            for key in staged_infos[0]:
                if key == "success":
                    staged_summary[key] = float(max(info[key] for info in staged_infos))
                else:
                    staged_summary[key] = float(sum(info[key] for info in staged_infos))

        segments.append({
            "obs": deepcopy(chunk_obs),
            "action_chunk": padded_actions,
            "action_mask": action_mask,
            "episode_id": episode_id,
            "timestep": timestep - len(executed_actions),
            "start_step": len(steps) - len(executed_actions),
            "end_step": len(steps),
            "segment_reward": float(sum(rewards)),
            "reward_seq": np.array(rewards, dtype=np.float32),
            "staged_summary": staged_summary,
            "done": bool(done),
            "success": bool(success or staged_summary.get("success", 0.0) > 0.0),
        })

        if done:
            break

    return {
        "episode_id": episode_id,
        "steps": steps,
        "segments": segments,
        "success": bool(success),
    }


def get_data_loaders(config, shape_meta):
    trainset, _ = TrainUtils.load_data_for_training(config, obs_keys=shape_meta["all_obs_keys"])
    obs_norm_stats = trainset.get_obs_normalization_stats() if config.train.hdf5_normalize_obs else None
    action_norm_stats = trainset.get_action_normalization_stats()
    return trainset, obs_norm_stats, action_norm_stats


def make_demo_loader(trainset, batch_size, num_workers):
    if batch_size <= 0:
        return None
    sampler = trainset.get_dataset_sampler()
    return DataLoader(
        dataset=trainset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=num_workers,
        drop_last=True,
    )


def train(config, device):
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    torch.set_num_threads(2)

    print("\n============= New Flow RWR Run with Config =============")
    print(config)
    print("")
    log_dir, ckpt_dir, video_dir, time_dir = TrainUtils.get_exp_dir(config, resume=False)
    latest_model_path = os.path.join(time_dir, "last.pth")
    latest_model_backup_path = os.path.join(time_dir, "last_bak.pth")

    if config.experiment.logging.terminal_output_to_txt:
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    ObsUtils.initialize_obs_utils_with_config(config)

    if isinstance(config.train.data, str):
        with config.values_unlocked():
            config.train.data = [{"path": config.train.data}]

    dataset_cfg = config.train.data[0]
    dataset_path = os.path.expanduser(dataset_cfg["path"])
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
    env_meta["lang"] = None
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_config=dataset_cfg,
        action_keys=config.train.action_keys,
        all_obs_keys=config.all_obs_keys,
        verbose=True,
    )

    ckpt_path = config.train.online.checkpoint_path or config.experiment.ckpt_path
    if ckpt_path is None:
        raise ValueError("must provide a pretrained flow checkpoint via train.online.checkpoint_path or experiment.ckpt_path")
    ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=ckpt_path)

    trainset, obs_normalization_stats, action_normalization_stats = get_data_loaders(config, shape_meta)
    model = build_model(config, shape_meta, device, ckpt_dict)

    demo_batch_ratio = float(config.train.online.demo_batch_ratio)
    total_batch_size = int(config.train.batch_size)
    demo_batch_size = int(round(total_batch_size * demo_batch_ratio))
    demo_batch_size = max(0, min(total_batch_size, demo_batch_size))
    online_batch_size = max(1, total_batch_size - demo_batch_size)
    demo_loader = make_demo_loader(trainset, demo_batch_size, config.train.num_data_workers) if demo_batch_size > 0 else None
    demo_iter = iter(demo_loader) if demo_loader is not None else None

    rollout_envs = create_envs(config, env_meta, shape_meta)
    train_env_name = next(iter(rollout_envs.keys()))
    train_env = rollout_envs[train_env_name]

    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )

    with open(os.path.join(log_dir, "..", "config.json"), "w") as outfile:
        json.dump(config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model)
    print("")
    print("*" * 50)
    flush_warnings()
    print("*" * 50)
    print("")

    best_success_rate = -1.0
    best_return = -np.inf
    rollout_policy = RolloutPolicy(
        model,
        obs_normalization_stats=obs_normalization_stats,
        action_normalization_stats=action_normalization_stats,
    )

    num_iters = int(config.train.online.num_iters)
    variable_state = {"iter": 0, "best_success_rate": best_success_rate, "best_return": best_return}
    for iter_idx in range(1, num_iters + 1):
        model.set_eval()
        buffer = FlowRolloutBuffer(
            action_horizon=model.algo_config.horizon.action_horizon,
            topk_fraction=float(config.algo.rwr.topk_fraction),
            use_segment_level_weighting=bool(config.algo.rwr.use_segment_level_weighting),
        )

        for episode_id in range(int(config.train.online.num_rollout_episodes_per_iter)):
            episode = collect_rollout_episode(
                env=train_env,
                rollout_policy=rollout_policy,
                model=model,
                episode_id=episode_id,
                horizon=int(config.train.online.rollout_horizon),
                terminate_on_success=bool(config.train.online.terminate_on_success),
            )
            buffer.add_episode(episode)

        buffer.compute_returns(gamma=float(config.algo.rwr.gamma))
        buffer.normalize_advantages(eps=float(config.algo.rwr.advantage_eps))
        buffer.compute_weights(
            temperature=float(config.algo.rwr.reward_temperature),
            min_weight=float(config.algo.rwr.min_weight),
            max_weight=float(config.algo.rwr.max_weight),
            topk_fraction=float(config.algo.rwr.topk_fraction),
        )
        rollout_stats = buffer.get_stats()

        model.set_train()
        train_logs = []
        for step in range(int(config.train.online.num_train_steps_per_iter)):
            demo_batch = None
            if demo_loader is not None:
                try:
                    demo_batch_raw = next(demo_iter)
                except StopIteration:
                    demo_iter = iter(demo_loader)
                    demo_batch_raw = next(demo_iter)
                demo_batch = model.process_batch_for_training(demo_batch_raw)
                demo_batch = model.postprocess_batch_for_training(
                    demo_batch,
                    obs_normalization_stats=obs_normalization_stats,
                )

            online_batch_raw = buffer.sample_batch(batch_size=online_batch_size, only_positive_weights=True)
            online_batch = model.process_online_batch_for_training(online_batch_raw)
            online_batch = model.postprocess_batch_for_training(
                online_batch,
                obs_normalization_stats=obs_normalization_stats,
            )

            info = model.train_on_mixed_batch(
                demo_batch=demo_batch,
                online_batch=online_batch,
                epoch=iter_idx,
                validate=False,
            )
            model.on_gradient_step()
            train_logs.append(model.log_info(info))

        mean_train_log = {}
        if len(train_logs) > 0:
            log_keys = train_logs[0].keys()
            for key in log_keys:
                mean_train_log[key] = float(np.mean([log[key] for log in train_logs]))

        print("Iter {}".format(iter_idx))
        print(json.dumps({
            "rollout": rollout_stats,
            "train": mean_train_log,
        }, sort_keys=True, indent=4))

        for key, value in rollout_stats.items():
            data_logger.record("Rollout/{}".format(key), value, iter_idx)
        for key, value in mean_train_log.items():
            data_logger.record("Train/{}".format(key), value, iter_idx)

        eval_interval = int(config.train.online.eval_interval)
        if eval_interval > 0 and (iter_idx % eval_interval == 0):
            model.set_eval()
            rollout_model = RolloutPolicy(
                model,
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )
            all_rollout_logs, _ = TrainUtils.rollout_with_stats(
                policy=rollout_model,
                envs=rollout_envs,
                horizon=int(config.train.online.rollout_horizon),
                use_goals=config.use_goals,
                num_episodes=int(config.train.online.num_eval_episodes),
                render=False,
                video_dir=video_dir if config.experiment.render_video else None,
                epoch=iter_idx,
                video_skip=config.experiment.get("video_skip", 5),
                terminate_on_success=bool(config.train.online.terminate_on_success),
            )
            for env_name, rollout_log in all_rollout_logs.items():
                for key, value in rollout_log.items():
                    record_key = "Eval/{}/{}".format(key, env_name)
                    data_logger.record(record_key, value, iter_idx, log_stats=True)
                if env_name == train_env_name:
                    best_success_rate = max(best_success_rate, rollout_log.get("Success_Rate", -1.0))
                    best_return = max(best_return, rollout_log.get("Return", -np.inf))

        variable_state = {
            "iter": iter_idx,
            "best_success_rate": best_success_rate,
            "best_return": best_return,
        }

        save_interval = int(config.train.online.save_interval)
        if save_interval > 0 and (iter_idx % save_interval == 0):
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                variable_state=variable_state,
                ckpt_path=os.path.join(ckpt_dir, "model_iter_{}.pth".format(iter_idx)),
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )

        print("\nsaving latest model at {}...\n".format(latest_model_path))
        TrainUtils.save_model(
            model=model,
            config=config,
            env_meta=env_meta,
            shape_meta=shape_meta,
            variable_state=variable_state,
            ckpt_path=latest_model_path,
            obs_normalization_stats=obs_normalization_stats,
            action_normalization_stats=action_normalization_stats,
        )
        shutil.copyfile(latest_model_path, latest_model_backup_path)
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, iter_idx)
        print("\nIter {} Memory Usage: {} MB\n".format(iter_idx, mem_usage))

    data_logger.close()


def main(args):
    if args.config is not None:
        ext_cfg = json.load(open(args.config, "r"))
        config = config_factory(ext_cfg["algo_name"])
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        config = config_factory(args.algo)

    if args.dataset is not None:
        config.train.data = [{"path": args.dataset}]
    if args.name is not None:
        config.experiment.name = args.name
    if args.ckpt_path is not None:
        config.train.online.checkpoint_path = args.ckpt_path

    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    if args.debug:
        config.unlock()
        config.lock_keys()
        config.train.online.num_iters = 2
        config.train.online.num_rollout_episodes_per_iter = 2
        config.train.online.num_train_steps_per_iter = 2
        config.train.online.rollout_horizon = 16
        config.train.online.eval_interval = 1
        config.train.online.save_interval = 1
        config.train.online.num_eval_episodes = 1
        config.train.output_dir = "/tmp/tmp_flow_rwr"

    config.lock()

    res_str = "finished run successfully!"
    try:
        train(config, device=device)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--algo", type=str, default="flow_rwr")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
