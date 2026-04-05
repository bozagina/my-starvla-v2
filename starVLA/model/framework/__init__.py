"""
Framework factory utilities.
Automatically builds registered framework implementations
based on configuration.

Each framework module (e.g., M1.py, QwenFast.py) should register itself:
    from starVLA.model.framework.framework_registry import FRAMEWORK_REGISTRY

    @FRAMEWORK_REGISTRY.register("InternVLA-M1")
    def build_model_framework(config):
        return InternVLA_M1(config=config)
"""

import pkgutil
import importlib
from starVLA.model.tools import FRAMEWORK_REGISTRY

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

LEGACY_FRAMEWORK_ALIASES = {
    # Legacy naming in some training yaml files.
    "QwenFM": "QwenGR00T",
}

try:
    pkg_path = __path__
except NameError:
    pkg_path = None

# Auto-import all framework submodules to trigger registration
if pkg_path is not None:
    for _, module_name, _ in pkgutil.iter_modules(pkg_path):
        try:
            importlib.import_module(f"{__name__}.{module_name}")
        except Exception as e:
            logger.warning(f"Failed to auto-import framework submodule {module_name}: {e}")


def _resolve_framework_id(cfg):
    if not hasattr(cfg.framework, "name"):
        # Backward compatibility for legacy config yaml.
        cfg.framework.name = cfg.framework.framework_py

    requested_id = cfg.framework.name
    resolved_id = LEGACY_FRAMEWORK_ALIASES.get(requested_id, requested_id)
    if resolved_id != requested_id:
        logger.warning(f"Framework `{requested_id}` is deprecated, fallback to `{resolved_id}`.")
    return requested_id, resolved_id


def build_framework(cfg):
    """
    Build a framework model from config.
    Args:
        cfg: Config object (OmegaConf / namespace) containing:
             cfg.framework.name: Identifier string (e.g. "InternVLA-M1")
    Returns:
        nn.Module: Instantiated framework model.
    """

    requested_id, framework_id = _resolve_framework_id(cfg)

    # auto detect from registry
    if framework_id not in FRAMEWORK_REGISTRY._registry:
        qwen_frameworks = sorted([name for name in FRAMEWORK_REGISTRY._registry if name.lower().startswith("qwen")])
        raise NotImplementedError(
            f"Framework `{requested_id}` (resolved `{framework_id}`) is not implemented. "
            f"Available Qwen frameworks: {qwen_frameworks}"
        )

    model_class = FRAMEWORK_REGISTRY[framework_id]
    return model_class(cfg)

__all__ = ["build_framework", "FRAMEWORK_REGISTRY"]
