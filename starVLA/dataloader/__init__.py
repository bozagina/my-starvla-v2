import json
import os
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist
from pathlib import Path

logger = get_logger(__name__)


def _inject_vla_builder_settings(cfg, vla_dataset_cfg):
    """Inject optional builder settings into vla_data config with backward-compatible fallbacks."""
    if vla_dataset_cfg is None:
        return
    if getattr(vla_dataset_cfg, "action_chunk_size", None) is not None:
        return

    action_model_cfg = getattr(getattr(cfg, "framework", None), "action_model", None)
    future_window = getattr(action_model_cfg, "future_action_window_size", None)
    if future_window is None:
        return
    try:
        action_chunk_size = int(future_window) + 1
    except Exception:
        return
    if action_chunk_size <= 0:
        return
    try:
        vla_dataset_cfg.action_chunk_size = action_chunk_size
    except Exception:
        logger.warning(
            "Unable to set `datasets.vla_data.action_chunk_size` dynamically; "
            "use explicit config if action window alignment is required."
        )

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"): # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data
        _inject_vla_builder_settings(cfg, vla_dataset_cfg)

        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        
        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            num_workers=4,
            # shuffle=True
        )        
        if dist.get_rank() == 0: 
            
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "vlm_datasets":
        from starVLA.dataloader.vlm_datasets import make_vlm_dataloader
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        
        return vlm_train_dataloader
