# Data Sources

## Primary Datasets

### UniRef Database
- **Source**: [UniProt UniRef](https://www.uniprot.org/help/uniref)
- **API**: `https://rest.uniprot.org/uniref/search`
- **Query**: `length:[* TO 50]` (short sequences)
- **Format**: FASTA
- **Access**: Public via UniProt REST API
- **Implementation**: `setup/prepare_data/prepare_uniref.py`

https://huggingface.co/datasets/nvidia/esm2_uniref_pretraining_data/viewer/default/train?row=4

### DBAASP (Database of Antimicrobial Activity and Structure of Peptides)
- **Source**: [DBAASP](https://dbaasp.org)
- **Version**: v3 (API 4.0.1)
- **Snapshot**: 2026-05-14
- **API**: [`https://dbaasp.org/peptides`](https://dbaasp.org/api?page=rest) (OpenAPI: `https://dbaasp.org/v3/api-docs`)
- **Query**: `complexity.value=monomer`, paginated via `limit`/`offset`, filtered client-side to `sequenceLength <= 50` and canonical AAs only
- **Format**: JSON
- **Access**: Public via DBAASP REST API; usage governed by the [DBAASP Terms and Conditions](https://dbaasp.org/docs/DBAASP_Terms_And_Conditions.pdf)
- **Implementation**: `setup/prepare_data/prepare_and_upload_dbaasp.py`

### dbAMP (Database of Antimicrobial Peptides)
- **Source**: [dbAMP](https://awi.cuhk.edu.cn/dbAMP/)
- **Version**: 3.0 (RELEASE 06/2024)
- **Files**: `data/dbamp/dbAMP_AntiGram_n_2024.fasta` (anti-Gram-negative), `data/dbamp/dbAMP_AntiGram_p_2024.fasta` (anti-Gram-positive)
- **Query**: union by `dbAMP_id`, filtered to `length <= 50` and canonical AAs only; activity preserved as `anti_gram_negative` / `anti_gram_positive` boolean columns
- **Format**: FASTA
- **Access**: Local file ingest (no network); usage governed by the dbAMP project license
- **Implementation**: `setup/prepare_data/prepare_and_upload_dbamp.py`

ampsphere
hydramp data
