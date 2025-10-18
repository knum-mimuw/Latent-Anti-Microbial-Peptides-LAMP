## LAMP data prep utilities

Create Hugging Face dataset repositories and upload prepared data as subsets.

### Setup
```bash
export HF_TOKEN=hf_xxx
```

### Commands

**Create dataset repository:**
```bash
uv run -m setup.prepare_data create_huggingface_dataset_repo --config setup/prepare_data/configs/create_huggingface_dataset_repo_config.yaml
```

**Prepare and upload ESM2 UniRef data:**
```bash
uv run -m setup.prepare_data prepare_and_upload_esm2_uniref --config setup/prepare_data/configs/prepare_and_upload_esm2_uniref_config.yaml
```

### Dataset Structure
```
pszmk/LAMP-datasets/
└── esm2-uniref/
    ├── train/
    └── validation/
```

### Workflow
1. Create the main dataset repository
2. Upload data as subsets (modify `subset_name` in config for different datasets)

### Help
```bash
uv run -m setup.prepare_data --help
```


