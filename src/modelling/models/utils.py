import torch
from typing import Any, Optional, Iterable
from pydantic import BaseModel
from typing import Dict


def get_obj_from_import_path(
    import_path: str, validation_prefix: Optional[str] = None
) -> Any:
    """Get an object from a import path."""
    module_name, obj_name = import_path.rsplit(".", 1)
    if validation_prefix and not obj_name.startswith(validation_prefix):
        raise ValueError(
            f"Object name {obj_name} does not start with {validation_prefix}. Available objects: {getattr(__import__(module_name), '__all__')}"
        )
    obj = getattr(__import__(module_name), obj_name)
    return obj


def load_model_from_huggingface(
    model_class_path: str,
    pretrained_model_name_or_path: Optional[str] = None,
    config_class_path: Optional[str] = None,
    load_pretrained: bool = True,
    config_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Load a model and optionally its config from Hugging Face.

    Args:
        model_class_path: Import path to the model class (e.g., 'transformers.AutoModel')
        pretrained_model_name_or_path: HuggingFace model identifier or local path
        config_class_path: Optional import path to config class (e.g., 'transformers.AutoConfig')
        load_pretrained: Whether to load pretrained weights. If False, instantiate from config.
        config_kwargs: Optional kwargs passed when constructing configs without pretrained weights.
        **kwargs: Additional arguments to pass to the model's from_pretrained method

    Returns:
        The loaded model instance

    Example:
        >>> model = load_model_from_huggingface(
        ...     model_class_path='transformers.AutoModel',
        ...     pretrained_model_name_or_path='bert-base-uncased'
        ... )
    """
    model_class = get_obj_from_import_path(model_class_path)
    config = None

    if config_class_path:
        config_class = get_obj_from_import_path(config_class_path)
        if pretrained_model_name_or_path:
            # Pull config from pretrained repository (weights optional)
            config = config_class.from_pretrained(
                pretrained_model_name_or_path, **(config_kwargs or {})
            )
        else:
            config = config_class(**(config_kwargs or {}))
    elif not load_pretrained and hasattr(model_class, "config_class"):
        config_cls = getattr(model_class, "config_class")
        if config_cls is not None:
            config = config_cls(**(config_kwargs or {}))

    if load_pretrained:
        if not pretrained_model_name_or_path:
            raise ValueError(
                "pretrained_model_name_or_path must be provided when load_pretrained is True"
            )
        if config is not None:
            return model_class.from_pretrained(
                pretrained_model_name_or_path, config=config, **kwargs
            )
        return model_class.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

    if config is None:
        raise ValueError(
            "Unable to instantiate model without pretrained weights. Provide config_class_path or config_kwargs."
        )

    return model_class(config)


class OptimizerConfig(BaseModel):
    optimizer_class_path: str
    optimizer_kwargs: Optional[Dict[str, Any]] = None


class SchedulerConfig(BaseModel):
    scheduler_class_path: str
    scheduler_kwargs: Optional[Dict[str, Any]] = None


def add_suffix_to_dict_keys(dict: Dict[str, Any], suffix: str) -> Dict[str, Any]:
    return {f"{key}_{suffix}": value for key, value in dict.items()}


def configure_optimizers(
    optimizer_config: OptimizerConfig,
    parameters: Iterable[torch.nn.Parameter],
    scheduler_config: Optional[SchedulerConfig] = None,
):
    optimizer_class = get_obj_from_import_path(optimizer_config.optimizer_class_path)
    optimizer = optimizer_class(parameters, **optimizer_config.optimizer_kwargs)
    if scheduler_config:
        scheduler_class = get_obj_from_import_path(
            scheduler_config.scheduler_class_path
        )
        scheduler = scheduler_class(optimizer, **scheduler_config.scheduler_kwargs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    else:
        return {"optimizer": optimizer}
