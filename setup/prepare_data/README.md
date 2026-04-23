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

**Prepare and upload APEX predictions sidecar data:**
```bash
uv run -m setup.prepare_data prepare_and_upload_apex_predictions --config setup/prepare_data/configs/prepare_and_upload_apex_predictions_config.yaml
```

**Prepare and upload strain condition sidecar data:**
```bash
uv run -m setup.prepare_data prepare_and_upload_strain_conditions --config setup/prepare_data/configs/prepare_and_upload_strain_conditions_config.yaml
```

**Prepare and upload physicochemical properties sidecar data:**
```bash
uv run -m setup.prepare_data prepare_and_upload_physicochemical_properties --config setup/prepare_data/configs/prepare_and_upload_physicochemical_properties_config.yaml
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
3. Generate and upload APEX prediction sidecar subset
4. Generate and upload strain-condition feature sidecar subset
5. Generate and upload physicochemical feature sidecar subset

### Help
```bash
uv run -m setup.prepare_data --help
```


