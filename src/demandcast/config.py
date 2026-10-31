import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    bundle_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")
    api_token: str = ""
    allow_colab_embed: bool = False

    @classmethod
    def from_env(cls):
        return cls(
            bundle_dir=Path(os.getenv("DC_BUNDLE_DIR", "artifacts")),
            reports_dir=Path(os.getenv("DC_REPORTS_DIR", "reports")),
            api_token=os.getenv("DC_API_TOKEN", ""),
            allow_colab_embed=os.getenv("DC_ALLOW_COLAB_EMBED", "").lower() == "true",
        )
