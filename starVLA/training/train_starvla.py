# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].


"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).  
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.  
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).  
"""

# Standard Library
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import re

# Third-Party Libraries
import torch
import torch.distributed as dist
import wandb
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups
from starVLA.training.trainer_utils.config_tracker import wrap_config, AccessTrackedConfig

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
from accelerate.logging import get_logger

logger = get_logger(__name__)


def _resolve_base_vlm_path(cfg):
    """Resolve base VLM path with Qwen-first preference while keeping legacy fallback."""
    fw = getattr(cfg, "framework", None)
    if fw is None:
        return None, None

    qwen_cfg = getattr(fw, "qwenvl", None)
    qwen_base = getattr(qwen_cfg, "base_vlm", None) if qwen_cfg is not None else None
    if qwen_base:
        return qwen_base, "framework.qwenvl.base_vlm"

    legacy_cfg = getattr(fw, "mapanything_llava3d", None)
    legacy_base = getattr(legacy_cfg, "base_vlm", None) if legacy_cfg is not None else None
    if legacy_base:
        return legacy_base, "framework.mapanything_llava3d.base_vlm"

    return None, None


def _resolve_vlm_interface(model):
    """Resolve runtime VLM interface attr on framework model (Qwen first, legacy fallback)."""
    for attr_name in ("qwen_vl_interface", "mapanythingllava3d_vlm_interface"):
        interface = getattr(model, attr_name, None)
        if interface is not None:
            return interface, attr_name
    return None, None


def _cfg_get(cfg_obj, key: str, default=None):
    if cfg_obj is None:
        return default
    if hasattr(cfg_obj, "get"):
        try:
            return cfg_obj.get(key, default)
        except Exception:
            pass
    return getattr(cfg_obj, key, default)


def _cfg_enabled(cfg_obj, key: str, default: bool = False) -> bool:
    value = _cfg_get(cfg_obj, key, default)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _to_int_or_none(value):
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _shape_2d(value):
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        dims = [int(x) for x in list(shape)]
    except Exception:
        return None
    if len(dims) != 2:
        return None
    if dims[0] <= 0 or dims[1] <= 0:
        return None
    return dims


def _ensure_single_process_deepspeed_env():
    """
    Ensure DeepSpeed has basic rank envs in single-process smoke runs.
    This avoids fallback MPI discovery (which requires `mpi4py`) when launcher
    env variables are missing.
    """
    os.environ.setdefault("DEEPSPEED_USE_MPI", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")


def build_accelerator(cfg) -> Accelerator:
    """Build accelerator with explicit gradient accumulation from config."""
    grad_accum_steps = int(getattr(cfg.trainer, "gradient_accumulation_steps", 1))
    if grad_accum_steps < 1:
        raise ValueError(f"Invalid gradient_accumulation_steps={grad_accum_steps}, must be >= 1")

    _ensure_single_process_deepspeed_env()

    deepspeed_plugin = None
    try:
        deepspeed_plugin = DeepSpeedPlugin(gradient_accumulation_steps=grad_accum_steps)
    except TypeError:
        # Backward compatibility for older accelerate versions.
        deepspeed_plugin = DeepSpeedPlugin()
        logger.warning(
            "DeepSpeedPlugin does not support `gradient_accumulation_steps` ctor arg in this environment; "
            "falling back to plugin defaults."
        )

    accelerator = Accelerator(
        deepspeed_plugin=deepspeed_plugin,
        gradient_accumulation_steps=grad_accum_steps,
    )
    accelerator.print(accelerator.state)
    accelerator.print(
        f"[accum] cfg.gradient_accumulation_steps={grad_accum_steps}, "
        f"accelerator.gradient_accumulation_steps={accelerator.gradient_accumulation_steps}"
    )
    return accelerator


def load_fast_tokenizer():
    fast_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
    return fast_tokenizer


def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        # create output directory and checkpoint directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

        # save config (both yaml and json for easy inspection)
        try:
            OmegaConf.save(cfg, output_dir / "config.yaml")
            with open(output_dir / "config.yaml", "r") as f_yaml, open(
                output_dir / "config.json", "w"
            ) as f_json:
                yaml_cfg = yaml.safe_load(f_yaml)
                json.dump(yaml_cfg, f_json, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save config to {output_dir}: {e}")

    return output_dir


def build_model(cfg) -> torch.nn.Module:
    """build model framework"""
    base_vlm, src_path = _resolve_base_vlm_path(cfg)
    if base_vlm:
        logger.info(f"Loading Base VLM `{base_vlm}` from {src_path}")
    else:
        logger.warning(
            "Unable to resolve base VLM path from config; expected `framework.qwenvl.base_vlm` "
            "or legacy `framework.mapanything_llava3d.base_vlm`."
        )
    model = build_framework(cfg)

    return model


# here changes need to 📦 encapsulate Dataloader
from starVLA.dataloader import build_dataloader


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader]:
    """prepare training data"""
    # VLA data loader
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_initialized():
        dist.barrier()

    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and scheduler"""
    # initialize optimizer
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    # print optimizer group info
    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,  # minimum learning rate
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        # training status tracking
        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        self.best_metric = None
        self.steps_since_improvement = 0
        trackers = getattr(self.config, "trackers", None)
        backend = None
        if trackers is not None:
            try:
                if "swanlab" in trackers:
                    backend = "swanlab"
                elif "wandb" in trackers:
                    backend = "wandb"
            except TypeError:
                backend = None
        if backend is None:
            backend = "wandb"
        self.logger_backend = backend
    
    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        cfg_grad_accum = int(getattr(self.config.trainer, "gradient_accumulation_steps", 1))
        actual_grad_accum = int(getattr(self.accelerator, "gradient_accumulation_steps", 1))
        if cfg_grad_accum != actual_grad_accum:
            raise RuntimeError(
                "Gradient accumulation mismatch detected: "
                f"cfg={cfg_grad_accum}, accelerator={actual_grad_accum}. "
                "Refuse to start training to avoid silent behavior drift."
            )

        # load pretrained weights
        self._init_checkpointing() # TODO merge with load pretrained weights

        # 根据  resume 调整 lr_scheduler
        self._adjust_lr_scheduler_for_resume()

        # freeze parameters
        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)

        #  print model trainable parameters:
        self.print_trainable_parameters(self.model)

        # initialize distributed training components
        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,  # must be the first param
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )
        self._patch_deepspeed_no_sync_if_needed()

        base_model = self.accelerator.unwrap_model(self.model)
        self._register_grad_hooks(base_model)

        self._init_wandb()

    def _patch_deepspeed_no_sync_if_needed(self):
        """
        Accelerate's `accumulate()` uses `no_sync` on non-sync micro steps.
        DeepSpeed ZeRO stage-2/3 with gradient partitioning disallows `no_sync`
        and raises:
          "no_sync context manager is incompatible with gradient partitioning ..."
        Replace instance-level `no_sync` with a no-op context manager to keep
        accumulation flow working.
        """
        dist_type = str(getattr(self.accelerator.state, "distributed_type", ""))
        if "DEEPSPEED" not in dist_type.upper():
            return
        engine = self.model
        if getattr(engine, "_starvla_no_sync_patched", False):
            return
        zero_partition_fn = getattr(engine, "zero_optimization_partition_gradients", None)
        if not callable(zero_partition_fn):
            return
        try:
            zero_partition = bool(zero_partition_fn())
        except Exception:
            zero_partition = False
        if not zero_partition:
            return

        @contextlib.contextmanager
        def _no_sync_passthrough():
            yield

        if hasattr(engine, "no_sync"):
            engine.no_sync = _no_sync_passthrough
            engine._starvla_no_sync_patched = True
            logger.warning(
                "Patched DeepSpeedEngine.no_sync -> nullcontext for ZeRO gradient partitioning "
                "compatibility (accumulate still works; gradients may sync each micro-step)."
            )


    def _adjust_lr_scheduler_for_resume(self):
        """根据已完成的步数调整学习率调度器状态"""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            
            # 方法1: 直接模拟已完成的步数（适用于大多数调度器）
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            
            # 或者方法2: 对于某些调度器，可以直接设置最后步数
            # if hasattr(self.lr_scheduler, '_step_count'):
            #     self.lr_scheduler._step_count = self.completed_steps
            
            logger.info(f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}")

    def _calculate_total_batch_size(self):
        """calculate global batch size"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """initialize Weights & Biases"""
        if not self.accelerator.is_main_process:
            return
        if getattr(self, "logger_backend", None) == "swanlab":
            import swanlab
            cfg_dict = OmegaConf.to_container(self.config, resolve=True)
            swanlab.init(
                project=self.config.wandb_project,
                workspace=self.config.wandb_entity,
                experiment_name=self.config.run_id,
                config=cfg_dict,
            )
        else:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 获取预训练检查点和是否恢复训练的标志
        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint
        # TODO retinking resume and load from pretrained_checkpoint
        if is_resume:
            # 恢复训练状态
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}")
                return None
            else:
                logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
                self.completed_steps = 0

        # 加载预训练权重
        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            try:
                self.completed_steps = int(re.search(r"steps_(\d+)_pytorch_model\.pt", pretrained_checkpoint).group(1))
            except AttributeError:
                logger.warning(f"Could not parse steps from pretrained checkpoint: {pretrained_checkpoint}")
                self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0
    

    def _load_checkpoint(self, checkpoint_path):
        """load checkpoint"""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """save current training state"""

        if self.accelerator.is_main_process:

            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            # save model state
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")

            # save training metadata
            summary_data = {
                "steps": self.completed_steps,
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")
            # ✅ Save accessed configuration only
            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                # self.config.save_accessed_config(
                #     output_dir / "config.json", 
                #     use_original_values=False
                # )
                self.config.save_accessed_config(
                    output_dir / "config.yaml", 
                    use_original_values=False 
                )
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """record training metrics"""
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if dist.get_rank() == 0:
                # add learning rate 
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0] # see lr group in yaml.trainer.learning_rate

                geom_vision_only_steps = getattr(self.config.trainer, "geom_vision_only_steps", 0)
                lang_freeze_steps = getattr(self.config.trainer, "lang_freeze_steps", 0)
                phase = 0
                if geom_vision_only_steps and self.completed_steps < geom_vision_only_steps:
                    phase = 1
                elif lang_freeze_steps and self.completed_steps < lang_freeze_steps:
                    phase = 2
                metrics["debug/training_phase"] = phase

                # add epoch info
                metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)

                # 1) 本地 JSONL 日志（方便离线 debug）
                try:
                    metrics_path = os.path.join(self.config.output_dir, "metrics.jsonl")
                    with open(metrics_path, "a") as f:
                        f.write(json.dumps(metrics) + "\n")
                except Exception as e:
                    logger.warning(f"Failed to write local metrics.jsonl: {e}")

                # 2) 远端日志（SwanLab / WandB）
                backend = getattr(self, "logger_backend", None)
                if backend == "swanlab":
                    try:
                        import swanlab
                        swanlab.log(metrics, step=self.completed_steps)
                    except Exception as e:
                        logger.warning(f"Failed to log metrics to SwanLab: {e}")
                elif backend == "wandb":
                    try:
                        wandb.log(metrics, step=self.completed_steps)
                    except Exception as e:
                        logger.warning(f"Failed to log metrics to WandB: {e}")

                # debug output
                logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """create data iterators"""
        self.vla_iter = iter(self.vla_train_dataloader)
        # self.vlm_iter = iter(self.vlm_train_dataloader)

    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """execute training loop"""
        # print training config
        self._log_training_config()

        # prepare data iterators
        self._create_data_iterators()
        self.optimizer.zero_grad()

        # create progress bar
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process
        )

        # main training loop
        while self.completed_steps < self.config.trainer.max_train_steps:
            # get data batch
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            # execute training step
            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            # update progress
            sync_step = bool(self.accelerator.sync_gradients)
            if sync_step:
                progress_bar.update(1)
                self.completed_steps += 1
            
            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                        {
                            "data_times": f"{t_end_data - t_start_data:.3f}",
                            "model_times": f"{t_end_model - t_start_model:.3f}",
                        }
                    )

            # Only run step-level side effects once per synchronized optimizer update.
            if not sync_step:
                continue

            # evaluate model
            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            # record metrics
            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            stop_training = False
            if not dist.is_initialized() or dist.get_rank() == 0:
                stop_training = self._check_early_stopping(step_metrics)

            if dist.is_initialized():
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                stop_flag = torch.tensor(int(stop_training), device=device)
                dist.broadcast(stop_flag, src=0)
                stop_training = bool(stop_flag.item())

            if stop_training:
                logger.info("Stopping training loop due to early stopping")
                break

            # save checkpoint
            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            # check termination condition
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        # training end processing
        self._finalize_training()

        # execute evaluation step

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """
        Evaluate the model on the given dataset using the specified metric function.

        :param eval_dataset: List of evaluation samples, each containing 'image', 'instruction', and 'action'.
        :param metric_fn: Function to compute the distance between predicted and ground truth actions.
        :return: Average metric score across the evaluation dataset.
        """

        examples = self._get_next_batch()
        score = 0.0
        num_samples = len(examples)
        actions = [example["action"] for example in examples]  # label
        # Predict actions using the model
        output_dict = self.model.predict_action(
            examples=examples, use_ddim=True, num_ddim_steps=20
        )

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]  # B, T, D
            actions = np.array(actions)  # convert actions to numpy.ndarray
            # B, Chunk, dim = actions.shape
            num_pots = np.prod(actions.shape)
            # Compute the metric score
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            average_score = score / num_pots
            step_metrics["mse_score"] = average_score

        del examples
        dist.barrier()  # ensure all processes are synchronized
        return step_metrics

    def _log_training_config(self):
        """record training config"""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps (cfg) = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(
                f"  Gradient accumulation steps (accelerator) = {self.accelerator.gradient_accumulation_steps}"
            )
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _register_grad_hooks(self, base_model=None):
        """register backward hooks on key modules to track output gradient norms (compatible with ZeRO)."""
        if base_model is None:
            base_model = self.model
        modules = []
        action_model = getattr(base_model, "action_model", None)
        if action_model is not None:
            dit = getattr(action_model, "model", None)
            if dit is not None:
                modules.append(dit)
            action_decoder = getattr(action_model, "action_decoder", None)
            if action_decoder is not None:
                modules.append(action_decoder)
        vlm_interface, _ = _resolve_vlm_interface(base_model)
        vlm_model = getattr(vlm_interface, "model", None) if vlm_interface is not None else None
        if vlm_model is not None:
            for name in [
                "geometric_projector",
                "fusion_projector",
                "vision_projector",
            ]:
                module = getattr(vlm_model, name, None)
                if module is not None:
                    modules.append(module)

        handles = []

        def make_hook():
            def hook(module, grad_input, grad_output):
                if not grad_output:
                    return
                out_grad = grad_output[0]
                if out_grad is None:
                    return
                g = out_grad.detach().float()
                if g.numel() == 0:
                    return
                sq = g * g
                module._last_grad_l2 = sq.sum().sqrt().item()
                module._last_grad_rms = sq.mean().sqrt().item()

            return hook

        for m in modules:
            try:
                h = m.register_full_backward_hook(make_hook())
                handles.append(h)
            except Exception:
                continue
        self._grad_hook_handles = handles

    def _check_early_stopping(self, metrics: dict) -> bool:
        es_cfg = getattr(self.config.trainer, "early_stopping", None)
        if not es_cfg or not getattr(es_cfg, "enabled", False):
            return False

        metric_name = getattr(es_cfg, "metric", "action_dit_loss")
        mode = getattr(es_cfg, "mode", "min")
        patience = getattr(es_cfg, "patience", 10000)
        min_delta = getattr(es_cfg, "min_delta", 0.0)

        if metric_name not in metrics:
            return False

        value = float(metrics[metric_name])

        if self.best_metric is None:
            self.best_metric = value
            self.steps_since_improvement = 0
            logger.info(f"Early stopping initialized on metric {metric_name} with value {value:.6f}")
            return False

        improved = (value < self.best_metric - min_delta) if mode == "min" else (value > self.best_metric + min_delta)

        if improved:
            self.best_metric = value
            self.steps_since_improvement = 0
            logger.info(f"Early stopping metric {metric_name} improved to {value:.6f}")
            return False

        self.steps_since_improvement += 1

        if self.steps_since_improvement >= patience:
            logger.info(
                f"Early stopping triggered on metric {metric_name}: "
                f"no improvement for {self.steps_since_improvement} steps"
            )
            return True

        return False
    
    def _collect_debug_norm_metrics(self, metrics: dict):
        """Collect parameter / gradient / weight stats for VLM, vision, geom and action model."""
        if not self.accelerator.is_main_process:
            return

        try:
            rank = dist.get_rank() if dist.is_initialized() else -1
        except Exception:
            rank = -1
        metrics["debug/grad_collect_called"] = 1
        metrics["debug/grad_collect_rank"] = int(rank)

        def module_norms(module):
            if module is None:
                return None, None
            param_sq = 0.0
            grad_sq = 0.0
            grad_count = 0
            for p in module.parameters():
                if not p.requires_grad:
                    continue
                if p.data is not None:
                    w = p.data.float()
                    param_sq += torch.sum(w * w).item()
                if p.grad is not None:
                    g = p.grad.detach().float()
                    grad_sq += torch.sum(g * g).item()
                    grad_count += 1
            if param_sq == 0.0 and grad_sq == 0.0:
                return None, None
            param_norm = param_sq**0.5 if param_sq > 0.0 else None
            hook_grad_l2 = getattr(module, "_last_grad_l2", None)
            if hook_grad_l2 is not None:
                grad_norm = float(hook_grad_l2)
            else:
                grad_norm = grad_sq**0.5 if grad_count > 0 else None
            return param_norm, grad_norm

        def _to_float_or_none(value):
            try:
                out = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(out):
                return None
            return out

        def module_grad_debug(name, module):
            info = {"n_params": 0, "n_grad": 0, "n_grad_nonzero": 0}
            if module is None:
                metrics[f"debug/grad_info/{name}_n_params"] = 0
                metrics[f"debug/grad_info/{name}_n_grad"] = 0
                metrics[f"debug/grad_info/{name}_n_grad_nonzero"] = 0
                metrics[f"debug/grad_info/{name}_hook_seen"] = 0.0
                metrics[f"debug/grad_info/{name}_hook_l2"] = 0.0
                metrics[f"debug/grad_info/{name}_hook_rms"] = 0.0
                metrics[f"debug/grad_info/{name}_zero_param_grad_but_hook"] = 0.0
                return
            for p in module.parameters():
                if not p.requires_grad:
                    continue
                info["n_params"] += 1
                if p.grad is not None:
                    info["n_grad"] += 1
                    with torch.no_grad():
                        v = p.grad.detach().float()
                        if v.abs().sum().item() > 0.0:
                            info["n_grad_nonzero"] += 1
            metrics[f"debug/grad_info/{name}_n_params"] = int(info["n_params"])
            metrics[f"debug/grad_info/{name}_n_grad"] = int(info["n_grad"])
            metrics[f"debug/grad_info/{name}_n_grad_nonzero"] = int(info["n_grad_nonzero"])
            hook_l2 = _to_float_or_none(getattr(module, "_last_grad_l2", None))
            hook_rms = _to_float_or_none(getattr(module, "_last_grad_rms", None))
            hook_seen = bool(hook_l2 is not None and math.isfinite(hook_l2))
            metrics[f"debug/grad_info/{name}_hook_seen"] = 1.0 if hook_seen else 0.0
            metrics[f"debug/grad_info/{name}_hook_l2"] = float(hook_l2) if hook_seen else 0.0
            metrics[f"debug/grad_info/{name}_hook_rms"] = (
                float(hook_rms) if (hook_rms is not None and math.isfinite(hook_rms)) else 0.0
            )
            metrics[f"debug/grad_info/{name}_zero_param_grad_but_hook"] = (
                1.0 if (info["n_grad"] == 0 and hook_seen) else 0.0
            )

        def weight_grad_stats(module):
            if module is None:
                return None, None, None, None
            param_sq = 0.0
            param_count = 0
            grad_sq = 0.0
            grad_count = 0
            for p in module.parameters():
                if not p.requires_grad:
                    continue
                if p.data is not None:
                    w = p.data.float()
                    param_sq += torch.sum(w * w).item()
                    param_count += w.numel()
                if p.grad is not None:
                    g = p.grad.detach().float()
                    grad_sq += torch.sum(g * g).item()
                    grad_count += g.numel()
            if param_count == 0 and grad_count == 0:
                return None, None, None, None
            w_l2 = param_sq**0.5 if param_sq > 0.0 else None
            w_rms = (param_sq / param_count)**0.5 if param_sq > 0.0 and param_count > 0 else None
            hook_grad_l2 = getattr(module, "_last_grad_l2", None)
            hook_grad_rms = getattr(module, "_last_grad_rms", None)
            if hook_grad_l2 is not None:
                g_l2 = float(hook_grad_l2)
            else:
                g_l2 = grad_sq**0.5 if grad_count > 0 else None
            if hook_grad_rms is not None:
                g_rms = float(hook_grad_rms)
            else:
                g_rms = (grad_sq / grad_count)**0.5 if grad_count > 0 else None
            return w_l2, w_rms, g_l2, g_rms

        try:
            base_model = self.accelerator.unwrap_model(self.model)
            action_model = getattr(base_model, "action_model", None)
            dit = getattr(action_model, "model", None) if action_model is not None else None
            action_decoder = getattr(action_model, "action_decoder", None) if action_model is not None else None

            vlm_interface, _ = _resolve_vlm_interface(base_model)
            vlm_model = getattr(vlm_interface, "model", None) if vlm_interface is not None else None
            vlm_language = getattr(vlm_model, "language_model", None) if vlm_model is not None else vlm_model

            module_grad_debug("dit", dit)
            dit_param, dit_grad = module_norms(dit)
            if dit_param is not None:
                metrics["debug/param_norm/dit"] = dit_param
            if dit_grad is not None:
                metrics["debug/grad_norm/dit"] = dit_grad

            module_grad_debug("action_decoder", action_decoder)
            dec_param, dec_grad = module_norms(action_decoder)
            if dec_param is not None:
                metrics["debug/param_norm/action_decoder"] = dec_param
            if dec_grad is not None:
                metrics["debug/grad_norm/action_decoder"] = dec_grad

            module_grad_debug("vlm_language", vlm_language)
            vlm_param, vlm_grad = module_norms(vlm_language)
            if vlm_param is not None:
                metrics["debug/param_norm/vlm_language"] = vlm_param
            if vlm_grad is not None:
                metrics["debug/grad_norm/vlm_language"] = vlm_grad

            if vlm_model is not None:
                geom_modules = [
                    ("geometric_model", getattr(vlm_model, "geometric_model", None)),
                    ("geometric_projector", getattr(vlm_model, "geometric_projector", None)),
                    ("fusion_projector", getattr(vlm_model, "fusion_projector", None)),
                ]
                for name, module in geom_modules:
                    module_grad_debug(name, module)
                    w_l2, w_rms, g_l2, g_rms = weight_grad_stats(module)
                    metrics[f"debug/geom_grad/{name}_l2"] = float(g_l2) if g_l2 is not None else 0.0
                    metrics[f"debug/geom_grad/{name}_rms"] = float(g_rms) if g_rms is not None else 0.0
                    metrics[f"debug/core_weight/{name}_w_rms"] = float(w_rms) if w_rms is not None else 0.0
                    if w_l2 is not None:
                        init_attr = "_param_l2_init"
                        last_attr = "_param_l2_last"
                        init_val = getattr(module, init_attr, None)
                        if init_val is None:
                            setattr(module, init_attr, w_l2)
                            setattr(module, last_attr, w_l2)
                            delta_from_start = 0.0
                            delta_from_last = 0.0
                        else:
                            last_val = getattr(module, last_attr, init_val)
                            delta_from_start = abs(w_l2 - init_val)
                            delta_from_last = abs(w_l2 - last_val)
                            setattr(module, last_attr, w_l2)
                        metrics[f"debug/param_delta/{name}_l2_from_start"] = float(delta_from_start)
                        metrics[f"debug/param_delta/{name}_l2_from_last"] = float(delta_from_last)

                vision_modules = [
                    ("vision_tower", getattr(vlm_model, "vision_tower", None)),
                    ("vision_projector", getattr(vlm_model, "vision_projector", None)),
                ]
                for name, module in vision_modules:
                    module_grad_debug(name, module)
                    w_l2, w_rms, g_l2, g_rms = weight_grad_stats(module)
                    metrics[f"debug/vlm_vision_grad/{name}_l2"] = float(g_l2) if g_l2 is not None else 0.0
                    metrics[f"debug/vlm_vision_grad/{name}_rms"] = float(g_rms) if g_rms is not None else 0.0
                    metrics[f"debug/core_weight/{name}_w_rms"] = float(w_rms) if w_rms is not None else 0.0
                    if w_l2 is not None:
                        init_attr = "_param_l2_init"
                        last_attr = "_param_l2_last"
                        init_val = getattr(module, init_attr, None)
                        if init_val is None:
                            setattr(module, init_attr, w_l2)
                            setattr(module, last_attr, w_l2)
                            delta_from_start = 0.0
                            delta_from_last = 0.0
                        else:
                            last_val = getattr(module, last_attr, init_val)
                            delta_from_start = abs(w_l2 - init_val)
                            delta_from_last = abs(w_l2 - last_val)
                            setattr(module, last_attr, w_l2)
                        metrics[f"debug/param_delta/{name}_l2_from_start"] = float(delta_from_start)
                        metrics[f"debug/param_delta/{name}_l2_from_last"] = float(delta_from_last)

                module_grad_debug("language_model", vlm_language)
                w_l2, w_rms, g_l2, g_rms = weight_grad_stats(vlm_language)
                metrics["debug/vlm_vision_grad/language_model_l2"] = float(g_l2) if g_l2 is not None else 0.0
                metrics["debug/vlm_vision_grad/language_model_rms"] = float(g_rms) if g_rms is not None else 0.0
                metrics["debug/core_weight/language_model_w_rms"] = float(w_rms) if w_rms is not None else 0.0
                if w_l2 is not None:
                    init_attr = "_param_l2_init"
                    last_attr = "_param_l2_last"
                    init_val = getattr(vlm_language, init_attr, None)
                    if init_val is None:
                        setattr(vlm_language, init_attr, w_l2)
                        setattr(vlm_language, last_attr, w_l2)
                        delta_from_start = 0.0
                        delta_from_last = 0.0
                    else:
                        last_val = getattr(vlm_language, last_attr, init_val)
                        delta_from_start = abs(w_l2 - init_val)
                        delta_from_last = abs(w_l2 - last_val)
                        setattr(vlm_language, last_attr, w_l2)
                    metrics["debug/param_delta/language_model_l2_from_start"] = float(delta_from_start)
                    metrics["debug/param_delta/language_model_l2_from_last"] = float(delta_from_last)
        except Exception as e:
            logger.warning(f"Failed to collect debug norm metrics: {e}")
            try:
                metrics["debug/grad_collect_error"] = str(e)
            except Exception:
                pass

    def _compute_optional_hook_losses(self, output_dict: dict, action_loss: torch.Tensor, step_metrics: dict):
        """Optionally blend extra task losses into total loss (default disabled)."""
        hooks_cfg = _cfg_get(getattr(self.config, "trainer", None), "optional_loss_hooks", None)
        if not _cfg_enabled(hooks_cfg, "enabled", default=False):
            return action_loss

        strict_missing_key = _cfg_enabled(hooks_cfg, "strict_missing_key", default=False)
        total_loss = action_loss
        hook_specs = [
            ("a_loss", "a_loss", "loss/a_module"),
            ("corrective_loss", "corrective_loss", "loss/corrective"),
        ]

        for hook_name, default_key, default_metric_name in hook_specs:
            hook_cfg = _cfg_get(hooks_cfg, hook_name, None)
            if not _cfg_enabled(hook_cfg, "enabled", default=False):
                continue

            hook_key = str(_cfg_get(hook_cfg, "key", default_key))
            metric_name = str(_cfg_get(hook_cfg, "metric_name", default_metric_name))
            scale = float(_cfg_get(hook_cfg, "scale", 1.0))
            hook_value = output_dict.get(hook_key, None)

            if hook_value is None:
                if strict_missing_key:
                    raise KeyError(
                        f"optional_loss_hooks requires key={hook_key!r} for {hook_name}, "
                        "but model forward output does not contain it."
                    )
                step_metrics[f"debug/{hook_name}_missing"] = 1.0
                continue

            if isinstance(hook_value, (float, int)):
                hook_loss = torch.tensor(
                    float(hook_value),
                    device=action_loss.device,
                    dtype=action_loss.dtype,
                )
            elif torch.is_tensor(hook_value):
                hook_loss = hook_value.to(device=action_loss.device, dtype=action_loss.dtype)
                if hook_loss.ndim > 0:
                    hook_loss = hook_loss.mean()
            else:
                raise TypeError(
                    f"optional_loss_hooks expects float/int/tensor for key={hook_key!r}, got {type(hook_value)}"
                )

            total_loss = total_loss + scale * hook_loss
            step_metrics[metric_name] = float(hook_loss.detach().float().item())
            step_metrics[f"debug/{hook_name}_scale"] = float(scale)

        step_metrics["loss/total"] = float(total_loss.detach().float().item())
        return total_loss

    def _check_shared_builder_contract(self, batch_vla, step_metrics: dict):
        """Optional contract assertions for shared builder samples. Default disabled."""
        contract_cfg = _cfg_get(getattr(self.config, "trainer", None), "shared_builder_contract_check", None)
        enabled = _cfg_enabled(contract_cfg, "enabled", default=False)
        if not enabled:
            return

        mode = str(_cfg_get(contract_cfg, "mode", "auto")).strip().lower()
        if mode not in {"auto", "shared", "legacy"}:
            raise ValueError(
                f"Invalid trainer.shared_builder_contract_check.mode={mode!r}; "
                "expected one of: auto/shared/legacy."
            )
        expected_schema_version = str(
            _cfg_get(contract_cfg, "expected_schema_version", "p1_shared_builder_v1")
        )
        expected_action_chunk_len = _to_int_or_none(
            _cfg_get(contract_cfg, "expected_action_chunk_len", None)
        )
        expected_action_dim = _to_int_or_none(_cfg_get(contract_cfg, "expected_action_dim", None))
        require_state = _cfg_enabled(contract_cfg, "require_state", default=False)
        strict_legacy_no_extra = _cfg_enabled(
            contract_cfg, "strict_legacy_no_extra", default=True
        )

        if not isinstance(batch_vla, list) or len(batch_vla) == 0:
            raise ValueError("shared builder contract check expects non-empty list batch.")
        sample = batch_vla[0]
        if not isinstance(sample, dict):
            raise ValueError(f"shared builder contract check expects dict sample, got {type(sample)}")

        sample_keys = set(sample.keys())
        resolved_mode = mode
        if resolved_mode == "auto":
            resolved_mode = (
                "shared"
                if all(k in sample_keys for k in ("obs", "action_chunk", "meta"))
                else "legacy"
            )

        errors = []
        base_required = {"action", "image", "lang"}
        missing_base = sorted(base_required - sample_keys)
        if missing_base:
            errors.append(f"missing base keys: {missing_base}")
        if require_state and "state" not in sample_keys:
            errors.append("`state` is required but missing.")

        action_shape = _shape_2d(sample.get("action"))
        if action_shape is None:
            errors.append("`action` must be rank-2 with positive shape [T, D].")
        else:
            if expected_action_chunk_len is not None and action_shape[0] != expected_action_chunk_len:
                errors.append(
                    f"`action` T mismatch: got {action_shape[0]}, expected {expected_action_chunk_len}."
                )
            if expected_action_dim is not None and action_shape[1] != expected_action_dim:
                errors.append(
                    f"`action` D mismatch: got {action_shape[1]}, expected {expected_action_dim}."
                )

        if resolved_mode == "shared":
            missing_shared = sorted({"obs", "action_chunk", "meta"} - sample_keys)
            if missing_shared:
                errors.append(f"shared mode missing keys: {missing_shared}")
            action_chunk_shape = _shape_2d(sample.get("action_chunk"))
            if action_chunk_shape is None:
                errors.append("shared mode requires `action_chunk` rank-2 [T, D].")
            elif action_shape is not None and action_shape != action_chunk_shape:
                errors.append(
                    f"`action_chunk` shape mismatch: got {action_chunk_shape}, expected {action_shape}."
                )

            meta = sample.get("meta")
            if not isinstance(meta, dict):
                errors.append("shared mode requires `meta` dict.")
            else:
                if meta.get("schema_version") != expected_schema_version:
                    errors.append(
                        f"`meta.schema_version` mismatch: got {meta.get('schema_version')!r}, "
                        f"expected {expected_schema_version!r}."
                    )
                for key in [
                    "dataset_name",
                    "trajectory_id",
                    "sample_step",
                    "action_keys",
                    "state_keys",
                    "video_keys",
                    "action_chunk_len",
                    "action_dim",
                ]:
                    if key not in meta:
                        errors.append(f"`meta.{key}` is required in shared mode.")

                if action_shape is not None:
                    if meta.get("action_chunk_len") != action_shape[0]:
                        errors.append(
                            f"`meta.action_chunk_len` mismatch: got {meta.get('action_chunk_len')}, expected {action_shape[0]}."
                        )
                    if meta.get("action_dim") != action_shape[1]:
                        errors.append(
                            f"`meta.action_dim` mismatch: got {meta.get('action_dim')}, expected {action_shape[1]}."
                        )
        elif strict_legacy_no_extra:
            forbidden = sorted(sample_keys.intersection({"obs", "action_chunk", "meta"}))
            if forbidden:
                errors.append(f"legacy mode contains shared-only keys: {forbidden}")

        step_metrics["debug/shared_builder_contract_checked"] = 1.0
        step_metrics["debug/shared_builder_contract_is_shared"] = (
            1.0 if resolved_mode == "shared" else 0.0
        )
        if errors:
            step_metrics["debug/shared_builder_contract_fail"] = 1.0
            raise ValueError("Shared-builder contract check failed: " + " | ".join(errors))

    def _train_step(self, batch_vla, batch_vlm=None):
        """execute single training step"""
        step_metrics = {}
        with self.accelerator.accumulate(self.model):
            self._check_shared_builder_contract(batch_vla, step_metrics)
            # VLA task forward propagation
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]

            debug_metrics = output_dict.get("debug_metrics", None)
            action_loss_value = float(action_loss.detach().float().item())
            step_metrics["action_dit_loss"] = action_loss_value
            step_metrics["loss/action"] = action_loss_value
            if isinstance(debug_metrics, dict):
                step_metrics.update(debug_metrics)
            total_loss = self._compute_optional_hook_losses(
                output_dict=output_dict,
                action_loss=action_loss,
                step_metrics=step_metrics,
            )
            if "loss/total" not in step_metrics:
                step_metrics["loss/total"] = float(total_loss.detach().float().item())

            # Guard against loss explosion before backward.
            if not torch.isfinite(total_loss).all():
                logger.warning(
                    "Detected non-finite loss at completed_steps=%d, skip optimizer update.",
                    self.completed_steps,
                )
                step_metrics["debug/nonfinite_loss"] = 1.0
                self.optimizer.zero_grad()
                return step_metrics

            # Smoke-only fast path: validate forward/loss contract without optimizer state allocation.
            if _cfg_enabled(getattr(self.config, "trainer", None), "smoke_forward_only", default=False):
                step_metrics["debug/smoke_forward_only"] = 1.0
                self.optimizer.zero_grad()
                return step_metrics

            # VLA backward propagation
            self.accelerator.backward(total_loss)

            geom_vision_only_steps = getattr(self.config.trainer, "geom_vision_only_steps", 0)
            lang_freeze_steps = getattr(self.config.trainer, "lang_freeze_steps", 0)
            if geom_vision_only_steps and self.completed_steps < geom_vision_only_steps:
                try:
                    base_model = self.accelerator.unwrap_model(self.model)
                    action_model = getattr(base_model, "action_model", None)
                    if action_model is not None:
                        for p in action_model.parameters():
                            if p.grad is not None:
                                p.grad.zero_()
                    vlm_interface, _ = _resolve_vlm_interface(base_model)
                    vlm_model = getattr(vlm_interface, "model", None) if vlm_interface is not None else None
                    language_model = getattr(vlm_model, "language_model", None) if vlm_model is not None else None
                    if language_model is not None:
                        for p in language_model.parameters():
                            if p.grad is not None:
                                p.grad.zero_()
                except Exception as e:
                    logger.warning(f"Failed to zero grads during geom_vision_only warmup: {e}")
            elif lang_freeze_steps and self.completed_steps < lang_freeze_steps:
                try:
                    base_model = self.accelerator.unwrap_model(self.model)
                    vlm_interface, _ = _resolve_vlm_interface(base_model)
                    vlm_model = getattr(vlm_interface, "model", None) if vlm_interface is not None else None
                    language_model = getattr(vlm_model, "language_model", None) if vlm_model is not None else None
                    if language_model is not None:
                        for p in language_model.parameters():
                            if p.grad is not None:
                                p.grad.zero_()
                except Exception as e:
                    logger.warning(f"Failed to zero language model grads during warmup: {e}")

            # gradient clipping
            clipped_grad_norm = None
            if self.config.trainer.gradient_clipping is not None:
                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.trainer.gradient_clipping,
                    )
                    if isinstance(grad_norm, torch.Tensor):
                        clipped_grad_norm = float(grad_norm.detach().float().cpu().item())
                    elif grad_norm is not None:
                        clipped_grad_norm = float(grad_norm)
                    if clipped_grad_norm is not None:
                        step_metrics["debug/clip_grad_norm"] = clipped_grad_norm

            # collect grad/weight stats before optimizer step (only on synchronized steps)
            if self.accelerator.sync_gradients:
                self._collect_debug_norm_metrics(step_metrics)

            # optimizer step
            if self.accelerator.sync_gradients:
                if clipped_grad_norm is not None and not math.isfinite(clipped_grad_norm):
                    logger.warning(
                        "Detected non-finite clipped grad norm at completed_steps=%d, skip optimizer update.",
                        self.completed_steps,
                    )
                    step_metrics["debug/nonfinite_grad_norm"] = 1.0
                    self.optimizer.zero_grad()
                    return step_metrics

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

        return step_metrics

    def _finalize_training(self):
        """training end processing"""
        # save final model
        if self.accelerator.is_main_process:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")


        # close W&B
        if self.accelerator.is_main_process:
            backend = getattr(self, "logger_backend", None)
            if backend == "swanlab":
                try:
                    import swanlab
                    swanlab.finish()
                except Exception:
                    pass
            elif backend == "wandb":
                wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    print("VLA Training :: Warming Up")

    #  Wrap config to enable access tracking
    cfg = wrap_config(cfg)
    accelerator = build_accelerator(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    # create output directory and save config
    output_dir = setup_directories(cfg=cfg)
    # build model
    vla = build_framework(cfg)
    # prepare data
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)

    # set optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    # create trainer
    # Run VLA Training
    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    # execute training preparation
    trainer.prepare_training()
    # execute training
    trainer.train()

    # And... we're done!
    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # if cfg.is_debug:
    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
