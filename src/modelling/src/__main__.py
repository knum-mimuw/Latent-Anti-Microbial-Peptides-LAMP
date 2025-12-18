"""Lightning CLI entry point for LAMP modelling package."""

from pytorch_lightning.cli import LightningCLI


def main():
    """Main entry point for Lightning CLI."""
    # Avoid clobbering the repository-level `config.yaml` by writing LightningCLI's
    # saved config to a dedicated filename.
    LightningCLI(
        run=True,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"config_filename": "lightning_cli_config.yaml", "overwrite": True},
    )


if __name__ == "__main__":
    main()
