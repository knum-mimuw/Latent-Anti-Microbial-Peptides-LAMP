"""Lightning CLI entry point for LAMP modelling package."""

from pytorch_lightning.cli import LightningCLI


def main():
    """Main entry point for Lightning CLI."""
    LightningCLI(run=True, subclass_mode_model=True, subclass_mode_data=True)


if __name__ == "__main__":
    main()
