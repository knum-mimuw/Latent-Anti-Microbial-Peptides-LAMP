from pytorch_lightning.cli import LightningCLI


def main() -> None:
    LightningCLI(run=True, subclass_mode_model=True, subclass_mode_data=True)


if __name__ == "__main__":
    main()
