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

ampsphere
hydramp data
