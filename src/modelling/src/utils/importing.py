import importlib
from typing import Any, Optional


def get_obj_from_import_path(
    import_path: str, validation_prefix: Optional[str] = None
) -> Any:
    """Get an object from a import path."""
    module_name, obj_name = import_path.rsplit(".", 1)
    if validation_prefix and not obj_name.startswith(validation_prefix):
        raise ValueError(
            f"Object name {obj_name} does not start with {validation_prefix}."
        )
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


def load_model_from_huggingface(
    model_class_path: str,
    pretrained_model_name_or_path: Optional[str] = None,
    config_class_path: Optional[str] = None,
    load_pretrained: bool = True,
    config_kwargs: Optional[dict] = None,
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
        return model_class.from_pretrained(pretrained_model_name_or_path, **kwargs)

    if config is None:
        raise ValueError(
            "Unable to instantiate model without pretrained weights. Provide config_class_path or config_kwargs."
        )

    return model_class(config)
