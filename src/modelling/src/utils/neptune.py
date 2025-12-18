from pathlib import Path
from typing import Any, Dict, Optional

from pytorch_lightning.loggers.logger import Logger
import yaml
import json

_PROJECT_ROOT_CACHE: Optional[Path] = None


def _project_root() -> Path:
    global _PROJECT_ROOT_CACHE
    if _PROJECT_ROOT_CACHE is not None:
        return _PROJECT_ROOT_CACHE

    here = Path(__file__).resolve()
    best: Optional[Path] = None
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            best = parent

    if best is not None:
        _PROJECT_ROOT_CACHE = best
        return best

    _PROJECT_ROOT_CACHE = here.parents[4]
    return _PROJECT_ROOT_CACHE


def _import_neptune() -> Any:
    try:
        import neptune  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Neptune is not installed. Install it with `pip install neptune` (or add it to your env) "
            "to enable Neptune logging and artifact downloads."
        ) from e
    return neptune


def _resolve_lightning_neptune_logger_cls() -> type:
    candidates = (
        "pytorch_lightning.loggers.neptune.NeptuneLogger",
        "lightning.pytorch.loggers.neptune.NeptuneLogger",
    )
    last_err: Optional[Exception] = None
    for dotted in candidates:
        module_name, cls_name = dotted.rsplit(".", 1)
        try:
            module = __import__(module_name, fromlist=[cls_name])
            return getattr(module, cls_name)
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise ImportError(
        "Could not import Lightning's NeptuneLogger. Ensure `pytorch-lightning` (or `lightning`) is installed."
    ) from last_err


class NeptuneLogger(Logger):
    def __init__(
        self,
        project: str,
        api_token: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._neptune_token = api_token
        self._neptune_project = project
        self._neptune_name = name
        self._neptune_kwargs = kwargs
        self._impl: Optional[Logger] = None

    def _get_impl(self) -> Logger:
        if self._impl is not None:
            return self._impl

        _import_neptune()
        impl_cls = _resolve_lightning_neptune_logger_cls()

        init_kwargs = dict(self._neptune_kwargs)
        if self._neptune_token is not None:
            init_kwargs.setdefault("api_token", self._neptune_token)
            init_kwargs.setdefault("api_key", self._neptune_token)

        self._impl = impl_cls(project=self._neptune_project, name=self._neptune_name, **init_kwargs)
        return self._impl

    @property
    def name(self) -> str:
        return self._get_impl().name

    @property
    def version(self) -> str:
        return str(self._get_impl().version)

    @property
    def experiment(self) -> Any:
        return self._get_impl().experiment

    def log_hyperparams(self, params: Any) -> None:
        self._get_impl().log_hyperparams(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        self._get_impl().log_metrics(metrics, step=step)

    def save(self) -> None:
        self._get_impl().save()

    def finalize(self, status: str) -> None:
        self._get_impl().finalize(status)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_impl(), name)


def download_neptune_config(
    run_id: str, project_name: str, api_token: str, output_path: Path | None = None
) -> Dict[str, Any]:
    """
    Download and load the config file from a Neptune experiment.

    Args:
        run_id: Neptune run ID
        project_name: Neptune project name
        api_token: Neptune API token
        output_path: Optional path to save the config. If None, saves in .neptune cache

    Returns:
        Dict containing the loaded config
    """
    # Set up cache directory if no output path specified
    if output_path is None:
        cache_dir = Path(".neptune") / project_name / run_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_path = cache_dir / "model_config.yaml"

    print(f"📄 Downloading config from run `{run_id}` to {output_path}")

    # Initialize Neptune run and download config
    neptune = _import_neptune()
    run = neptune.init_run(with_id=run_id, project=project_name, api_token=api_token)
    run["config/model_config.yaml"].download(destination=str(output_path))
    run.stop()
    print("✅ Config download complete")

    # Load and return the config
    with output_path.open() as f:
        if output_path.suffix.lower() in {".yml", ".yaml"}:
            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    return config


def download_checkpoint(
    run_id: str,
    checkpoint_name: str,
    neptune_project: str,
    api_token: str,
) -> Path | None:
    """
    Download a checkpoint from Neptune if it doesn't exist locally.

    Args:
        run_id: Neptune run ID
        checkpoint_name: Name of the checkpoint to download
        neptune_project: Neptune project name
        api_token: Neptune API token

    Returns:
        Path to the downloaded checkpoint or None if download failed
    """
    cache_dir = _project_root() / ".neptune" / neptune_project / run_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    auto_ckpt = cache_dir / f"{checkpoint_name}.ckpt"
    print(f"📥 Checkpoint path: {auto_ckpt}")

    if not auto_ckpt.exists():
        print(f"    → Downloading `{checkpoint_name}` from run `{run_id}`…")
        neptune = _import_neptune()
        run = neptune.init_run(
            with_id=run_id, project=neptune_project, api_token=api_token
        )
        run[f"training/model/checkpoints/{checkpoint_name}"].download(
            destination=str(auto_ckpt)
        )
        run.stop()
        print("    → Download complete.")
    else:
        print("    → Already cached, skipping download.")

    return auto_ckpt if auto_ckpt.exists() else None
