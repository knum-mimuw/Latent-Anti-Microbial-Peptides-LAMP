## LAMP data prep utilities

Create Hugging Face dataset repositories and upload prepared data as subsets.

### Setup
```bash
export HF_TOKEN=hf_xxx
```

### Commands

**Create dataset repository:**
```bash
uv run -m setup.prepare_data create_huggingface_dataset_repo --config setup/prepare_data/configs/repository/create_huggingface_dataset_repo_config.yaml
```

**Prepare and upload ESM2 UniRef data:**
```bash
uv run -m setup.prepare_data prepare_and_upload_esm2_uniref --config setup/prepare_data/configs/esm2_uniref/prepare_and_upload_esm2_uniref_config.yaml
```

**Prepare and upload DBAASP peptides:**
```bash
uv run -m setup.prepare_data prepare_and_upload_dbaasp --config setup/prepare_data/configs/dbaasp/prepare_and_upload_dbaasp_config.yaml
```

**Prepare and upload dbAMP peptides:**
```bash
uv run -m setup.prepare_data prepare_and_upload_dbamp --config setup/prepare_data/configs/dbamp/prepare_and_upload_dbamp_config.yaml
```

**Prepare and upload APEX predictions sidecar data:**
```bash
uv run -m setup.prepare_data prepare_and_upload_apex_predictions --config setup/prepare_data/configs/apex_predictions/prepare_and_upload_apex_predictions_config.yaml
```

**Prepare and upload strain condition sidecar data:**
```bash
uv run -m setup.prepare_data prepare_and_upload_strain_conditions --config setup/prepare_data/configs/strain_conditions/prepare_and_upload_strain_conditions_config.yaml
```

**Prepare and upload physicochemical properties sidecar data:**
```bash
uv run -m setup.prepare_data prepare_and_upload_physicochemical_properties --config setup/prepare_data/configs/physicochemical_properties/prepare_and_upload_physicochemical_properties_config.yaml
```

### Dataset Structure
```
pszmk/LAMP-datasets/
├── nvidia_esm2_uniref_pretraining_data_leq50aa/
│   ├── train/
│   └── validation/
├── dbaasp_peptides_leq50aa/
│   └── train/
└── dbamp_peptides_leq50aa/
    └── train/
```

### Source data

| Source | Version | Provenance |
| --- | --- | --- |
| DBAASP | v3 (API 4.0.1), snapshot 2026-05-14 | REST ingest from `https://dbaasp.org/peptides` (OpenAPI: `https://dbaasp.org/v3/api-docs`). Server filter `complexity.value=monomer`, paginated 500 records at a time. 24,397 monomer records pre-filter, 19,202 retained after `length <= 50` + canonical-AA filter into `dbaasp_peptides_leq50aa`. Usage governed by the [DBAASP Terms and Conditions](https://dbaasp.org/docs/DBAASP_Terms_And_Conditions.pdf). |
| dbAMP | v3.0 (RELEASE 06/2024) | Two FASTA files in `data/dbamp/` (`dbAMP_AntiGram_n_2024.fasta`, `dbAMP_AntiGram_p_2024.fasta`) from the [dbAMP project](https://awi.cuhk.edu.cn/dbAMP/). Unioned by `dbAMP_id` into `dbamp_peptides_leq50aa` with `anti_gram_negative` and `anti_gram_positive` boolean labels. |

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


