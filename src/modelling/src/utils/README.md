# Utils Module

Shared utility functions used across the modelling package.

## Functions

### `get_obj_from_import_path`

Dynamically loads objects (classes, functions) from import paths.

```python
from utils.importing import get_obj_from_import_path

# Load a class
ModelClass = get_obj_from_import_path("torch.nn.Linear")
model = ModelClass(10, 5)

# Load a function
loss_fn = get_obj_from_import_path("torch.nn.functional.mse_loss")
```

**Parameters:**
- `import_path`: Full import path (e.g., `"torch.nn.Linear"`)
- `validation_prefix`: Optional prefix to validate object name against

**Returns:** The imported object (class, function, etc.)

### `load_model_from_huggingface`

Loads models from Hugging Face with optional config handling.

```python
from utils.importing import load_model_from_huggingface

# Load pretrained model
model = load_model_from_huggingface(
    model_class_path="transformers.AutoModel",
    pretrained_model_name_or_path="bert-base-uncased"
)

# Load model from config
model = load_model_from_huggingface(
    model_class_path="transformers.AutoModel",
    config_class_path="transformers.AutoConfig",
    load_pretrained=False,
    config_kwargs={"vocab_size": 1000, "hidden_size": 768}
)
```

**Parameters:**
- `model_class_path`: Import path to model class
- `pretrained_model_name_or_path`: HuggingFace model identifier or local path
- `config_class_path`: Optional import path to config class
- `load_pretrained`: Whether to load pretrained weights (default: True)
- `config_kwargs`: Optional kwargs for config when not loading pretrained
- `**kwargs`: Additional arguments passed to `from_pretrained`

**Returns:** The loaded model instance

## Usage

These utilities are used throughout the codebase for:
- Dynamic model loading in `MetaModule`
- Dynamic loss/metric loading in `LossManager` and `MetricsCallback`
- Hugging Face model integration in model implementations

