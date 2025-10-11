# Data Sources

## Primary Datasets

### UniRef Database
- **Source**: [UniProt UniRef](https://www.uniprot.org/help/uniref)
- **API**: `https://rest.uniprot.org/uniref/search`
- **Query**: `length:[* TO 50]` (short sequences)
- **Format**: FASTA
- **Access**: Public via UniProt REST API
- **Implementation**: `setup/prepare_data/prepare_uniref.py`

### Hugging Face Datasets
- **Platform**: [Hugging Face Hub](https://huggingface.co/datasets)
- **Purpose**: Dataset storage and sharing
- **Format**: Hugging Face Dataset format
- **Authentication**: HF_TOKEN environment variable
- **Implementation**: `setup/prepare_data/upload_data_to_huggingface.py`

## Secondary Sources
- Literature-curated AMP datasets (referenced in literature review)
- Experimental validation data
- Cross-references from multiple databases
