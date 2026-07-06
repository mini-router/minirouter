from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if str(path) and path.exists() and path.is_file():
            return path
    return None


DEFAULT_DATABASE_URL = "sqlite:///./data/eval_backend.db"
DEFAULT_ARTIFACT_ROOT = Path("./data/artifacts")
DEFAULT_WORKSPACE_ROOT = Path("./data/workspaces")
DEFAULT_LOCAL_REPO_DIR = Path(__file__).resolve().parents[4]
DEFAULT_ALLOWED_ORIGINS = ["http://localhost:5173"]
DEFAULT_GITHUB_WEBHOOK_SECRET = "replace-me"
DEFAULT_GITHUB_ACCESS_TOKEN = ""
DEFAULT_ALLOWED_REPO = "mini-router/minirouter"
DEFAULT_MINER_REPO_URL = "https://github.com/mini-router/minirouter"
DEFAULT_TRINITY_REMOTE_DIR = "trinity"
DEFAULT_TRINITY_REMOTE_WORKSPACE_ROOT = "~/trinity-eval-workspaces"
DEFAULT_TRINITY_GPU_INDEX = 5
DEFAULT_REMOTE_EVAL_COMMAND_TEMPLATE = (
    "PYTHONPATH=src "
    "python -m trinity.eval "
    "--benchmark {benchmark} --provider {provider} --models {models_config} "
    "--device cuda:0 --dtype bfloat16 --max-items 20 "
    "--theta {checkpoint_path} --out {results_path}"
)
DEFAULT_LOCAL_EVAL_COMMAND_TEMPLATE = (
    "PYTHONPATH=src "
    "python -m trinity.eval "
    "--benchmark {benchmark} --provider {provider} --models {models_config} "
    "--device cpu --dtype float32 --max-items 20 "
    "--theta {checkpoint_path} --out {results_path}"
)
DEFAULT_EVAL_PROVIDER = "openrouter"
DEFAULT_EVAL_MODELS_CONFIG = "configs/models.openrouter.yaml"
DEFAULT_EVAL_RESULT_POINTER = "results.TRINITY"
DEFAULT_EVAL_MAX_ITEMS = 20
DEFAULT_EVAL_BENCHMARK = "math500"
DEFAULT_GIT_AUTHOR_NAME = "Minirouter Evaluator"
DEFAULT_GIT_AUTHOR_EMAIL = "eval-bot@example.com"
DEFAULT_TRINITY_SECRETS_FILE = "./secrets.env"
DEFAULT_EVAL_TIMEOUT_SECONDS = 1800
DEFAULT_EVAL_EXECUTION_MODE = "remote_gpu"


@dataclass(slots=True)
class Settings:
    database_url: str = DEFAULT_DATABASE_URL
    artifact_root: Path = field(default_factory=lambda: DEFAULT_ARTIFACT_ROOT)
    workspace_root: Path = field(default_factory=lambda: DEFAULT_WORKSPACE_ROOT)
    allowed_origins: list[str] = field(default_factory=lambda: list(DEFAULT_ALLOWED_ORIGINS))
    github_webhook_secret: str = DEFAULT_GITHUB_WEBHOOK_SECRET
    github_access_token: str = DEFAULT_GITHUB_ACCESS_TOKEN
    allowed_repo: str = DEFAULT_ALLOWED_REPO
    miner_repo_url: str = DEFAULT_MINER_REPO_URL
    local_repo_dir: Path = field(default_factory=lambda: DEFAULT_LOCAL_REPO_DIR)
    trinity_remote_host: str = "trinity-gpu"
    trinity_remote_dir: str = DEFAULT_TRINITY_REMOTE_DIR
    trinity_remote_workspace_root: str = DEFAULT_TRINITY_REMOTE_WORKSPACE_ROOT
    trinity_gpu_index: int = DEFAULT_TRINITY_GPU_INDEX
    remote_eval_command_template: str = DEFAULT_REMOTE_EVAL_COMMAND_TEMPLATE
    local_eval_command_template: str = DEFAULT_LOCAL_EVAL_COMMAND_TEMPLATE
    eval_provider: str = DEFAULT_EVAL_PROVIDER
    eval_models_config: str = DEFAULT_EVAL_MODELS_CONFIG
    eval_result_pointer: str = DEFAULT_EVAL_RESULT_POINTER
    eval_max_items: int = DEFAULT_EVAL_MAX_ITEMS
    eval_benchmark: str = DEFAULT_EVAL_BENCHMARK
    git_author_name: str = DEFAULT_GIT_AUTHOR_NAME
    git_author_email: str = DEFAULT_GIT_AUTHOR_EMAIL
    trinity_secrets_file: str = DEFAULT_TRINITY_SECRETS_FILE
    eval_timeout_seconds: int = DEFAULT_EVAL_TIMEOUT_SECONDS
    eval_execution_mode: str = DEFAULT_EVAL_EXECUTION_MODE
    sync_eval_on_submit: bool = False

    @classmethod
    def load(cls) -> "Settings":
        root = _repo_root()
        candidate_files = [Path.cwd() / "secrets.env", root / "secrets.env"]
        trinity_secrets = os.environ.get("TRINITY_SECRETS_FILE", "").strip()
        if trinity_secrets:
            candidate_files.append(Path(trinity_secrets))
        env_path = _first_existing(candidate_files)
        file_values = _parse_env_file(env_path) if env_path else {}

        def get(name: str, default: str) -> str:
            return os.environ.get(name, file_values.get(name, default))

        origins = [
            origin.strip()
            for origin in get("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
            if origin.strip()
        ]

        return cls(
            database_url=get("DATABASE_URL", DEFAULT_DATABASE_URL),
            artifact_root=Path(get("ARTIFACT_ROOT", str(DEFAULT_ARTIFACT_ROOT))),
            workspace_root=Path(get("WORKSPACE_ROOT", str(DEFAULT_WORKSPACE_ROOT))),
            allowed_origins=origins,
            github_webhook_secret=get("GITHUB_WEBHOOK_SECRET", DEFAULT_GITHUB_WEBHOOK_SECRET),
            github_access_token=get("GITHUB_ACCESS_TOKEN", DEFAULT_GITHUB_ACCESS_TOKEN),
            allowed_repo=get("ALLOWED_REPO", DEFAULT_ALLOWED_REPO),
            miner_repo_url=get("MINER_REPO_URL", DEFAULT_MINER_REPO_URL),
            local_repo_dir=Path(get("MINIROUTER_REPO_DIR", str(DEFAULT_LOCAL_REPO_DIR))),
            trinity_remote_host=get("TRINITY_GPU_HOST", "trinity-gpu"),
            trinity_remote_dir=get("TRINITY_REMOTE_DIR", DEFAULT_TRINITY_REMOTE_DIR),
            trinity_remote_workspace_root=get(
                "TRINITY_REMOTE_WORKSPACE_ROOT", DEFAULT_TRINITY_REMOTE_WORKSPACE_ROOT
            ),
            trinity_gpu_index=int(get("TRINITY_GPU_INDEX", str(DEFAULT_TRINITY_GPU_INDEX))),
            remote_eval_command_template=get(
                "REMOTE_EVAL_COMMAND_TEMPLATE", DEFAULT_REMOTE_EVAL_COMMAND_TEMPLATE
            ),
            local_eval_command_template=get(
                "LOCAL_EVAL_COMMAND_TEMPLATE", DEFAULT_LOCAL_EVAL_COMMAND_TEMPLATE
            ),
            eval_provider=get("EVAL_PROVIDER", DEFAULT_EVAL_PROVIDER),
            eval_models_config=get("EVAL_MODELS_CONFIG", DEFAULT_EVAL_MODELS_CONFIG),
            eval_result_pointer=get("EVAL_RESULT_POINTER", DEFAULT_EVAL_RESULT_POINTER),
            eval_max_items=int(get("EVAL_MAX_ITEMS", str(DEFAULT_EVAL_MAX_ITEMS))),
            eval_benchmark=get("EVAL_BENCHMARK", DEFAULT_EVAL_BENCHMARK),
            git_author_name=get("GIT_AUTHOR_NAME", DEFAULT_GIT_AUTHOR_NAME),
            git_author_email=get("GIT_AUTHOR_EMAIL", DEFAULT_GIT_AUTHOR_EMAIL),
            trinity_secrets_file=get("TRINITY_SECRETS_FILE", DEFAULT_TRINITY_SECRETS_FILE),
            eval_timeout_seconds=int(get("EVAL_TIMEOUT_SECONDS", str(DEFAULT_EVAL_TIMEOUT_SECONDS))),
            eval_execution_mode=get("EVAL_EXECUTION_MODE", DEFAULT_EVAL_EXECUTION_MODE),
            sync_eval_on_submit=get("SYNC_EVAL_ON_SUBMIT", "false").lower()
            in {"1", "true", "yes", "on"},
        )

    def ensure_dirs(self) -> None:
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        (self.artifact_root / "uploads").mkdir(parents=True, exist_ok=True)
        (self.artifact_root / "extracted").mkdir(parents=True, exist_ok=True)
        (self.workspace_root / "submissions").mkdir(parents=True, exist_ok=True)
