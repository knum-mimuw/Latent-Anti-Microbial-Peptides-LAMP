"""Original HydrAMP publication (cite when referring to the HydrAMP architecture)."""

_TITLE = (
    "Discovering highly potent antimicrobial peptides with deep generative model {HydrAMP}"
)

_ABSTRACT = (
    "Antimicrobial peptides emerge as compounds that can alleviate the global health hazard "
    "of antimicrobial resistance, prompting a need for novel computational approaches to "
    "peptide generation. Here, we propose HydrAMP, a conditional variational autoencoder that "
    "learns lower-dimensional, continuous representation of peptides and captures their "
    "antimicrobial properties. The model disentangles the learnt representation of a peptide "
    "from its antimicrobial conditions and leverages parameter-controlled creativity. HydrAMP "
    "is the first model that is directly optimized for diverse tasks, including unconstrained "
    "and analogue generation and outperforms other approaches in these tasks. An additional "
    "preselection procedure based on ranking of generated peptides and molecular dynamics "
    "simulations increases experimental validation rate. Wet-lab experiments on five "
    "bacterial strains confirm high activity of nine peptides generated as analogues of "
    "clinically relevant prototypes, as well as six analogues of an inactive peptide. HydrAMP "
    "enables generation of diverse and potent peptides, making a step towards resolving the "
    "antimicrobial resistance crisis."
)

_AUTHOR = (
    "Szymczak, Paulina and Możejko, Marcin and Grzegorzek, Tomasz and Jurczak, Radosław and "
    "Bauer, Marta and Neubauer, Damian and Sikora, Karol and Michalski, Michał and Sroka, "
    "Jacek and Setny, Piotr and Kamysz, Wojciech and Szczurek, Ewa"
)

HYDRAMP_ORIGINAL_ARTICLE_BIBTEX = "\n".join(
    [
        "@article{szymczak_discovering_2023,",
        f"  title = {{{_TITLE}}},",
        "  volume = {14},",
        "  issn = {2041-1723},",
        "  url = {https://www.nature.com/articles/s41467-023-36994-z},",
        "  doi = {10.1038/s41467-023-36994-z},",
        f"  abstract = {{{_ABSTRACT}}},",
        "  language = {en},",
        "  number = {1},",
        "  journal = {Nature Communications},",
        f"  author = {{{_AUTHOR}}},",
        "  month = mar,",
        "  year = {2023},",
        "  keywords = {Computational models, Machine learning, Protein design},",
        "  pages = {1453},",
        "}",
    ]
)

HYDRAMP_README_CITATION_SECTION = (
    "## Citation\n\n"
    "The **HydrAMP** architecture and original model were introduced by Szymczak *et al.* "
    "in *Nature Communications* (2023). When you refer to HydrAMP or build on this work, "
    "please cite:\n\n"
    "```bibtex\n"
    + HYDRAMP_ORIGINAL_ARTICLE_BIBTEX.strip()
    + "\n```\n\n"
    "- **DOI:** [10.1038/s41467-023-36994-z](https://doi.org/10.1038/s41467-023-36994-z)\n"
    "- **Article:** [nature.com](https://www.nature.com/articles/s41467-023-36994-z)\n"
)
