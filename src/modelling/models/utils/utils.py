import torch
from typing import Any, Optional, Iterable


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
    pretrained_model_name_or_path: str,
    config_class_path: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Load a model and optionally its config from Hugging Face.

    Args:
        model_class_path: Import path to the model class (e.g., 'transformers.AutoModel')
        pretrained_model_name_or_path: HuggingFace model identifier or local path
        config_class_path: Optional import path to config class (e.g., 'transformers.AutoConfig')
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

    if config_class_path:
        config_class = get_obj_from_import_path(config_class_path)
        config = config_class.from_pretrained(pretrained_model_name_or_path)
        model = model_class.from_pretrained(
            pretrained_model_name_or_path, config=config, **kwargs
        )
    else:
        model = model_class.from_pretrained(pretrained_model_name_or_path, **kwargs)

    return model
