def _resolve_vlm_name(config):
    """Resolve VLM name/path from config (Qwen-first, legacy fallback)."""
    fw = getattr(config, "framework", None)
    if fw is None:
        return None

    qwen_base = getattr(getattr(fw, "qwenvl", None), "base_vlm", None)
    if qwen_base:
        return str(qwen_base)

    legacy_base = getattr(getattr(fw, "mapanything_llava3d", None), "base_vlm", None)
    if legacy_base:
        return str(legacy_base)

    return None


def get_vlm_model(config):
    vlm_name = _resolve_vlm_name(config)
    if not vlm_name:
        raise ValueError(
            "Cannot resolve base VLM path from config: expected `framework.qwenvl.base_vlm` "
            "or legacy `framework.mapanything_llava3d.base_vlm`."
        )

    vlm_name_l = vlm_name.lower()
    if (
        "qwen2.5-vl" in vlm_name_l
        or "qwen2.5" in vlm_name_l
        or "qwen2_5" in vlm_name_l
        or "qwen2-5" in vlm_name_l
        or "nora" in vlm_name_l
    ):
        from .QWen2_5 import _QWen_VL_Interface

        return _QWen_VL_Interface(config)
    if "qwen3-vl" in vlm_name_l or "qwen3vl" in vlm_name_l:
        from .QWen3 import _QWen3_VL_Interface

        return _QWen3_VL_Interface(config)
    if "florence" in vlm_name_l:
        from .Florence2 import _Florence_Interface

        return _Florence_Interface(config)
    if "mapanything_llava3d" in vlm_name_l or "mapanythingllava3d" in vlm_name_l:
        from .MapAnythingLlava3D import _MapAnythingLlava3D_Interface

        return _MapAnythingLlava3D_Interface(config)
    raise NotImplementedError(f"VLM model {vlm_name} not implemented")
