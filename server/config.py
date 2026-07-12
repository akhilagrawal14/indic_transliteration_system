"""Runtime configuration loaded from environment variables (.env)."""

from pydantic_settings import BaseSettings, SettingsConfigDict

from server.engine.base import validate_beam


class Settings(BaseSettings):
    """Server settings, read from XLIT_-prefixed env vars (and .env).

    The XLIT_ prefix is required, not cosmetic: an unprefixed `lang` field maps
    to the env var `LANG`, which collides with the POSIX locale variable (e.g.
    `LANG=C.UTF-8`) and silently overrides the intended value. Prefixing every
    field prevents that whole class of collision (LANG, PORT, ...).
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="XLIT_", extra="ignore"
    )

    engine: str = "ct2"                 # ct2 | fairseq
    device: str = "cpu"                 # cpu | cuda (cuda for benchmarking only)
    lang: str = "hi"
    beam_width: int = 5
    topk: int = 5

    dict_path: str = "server/data/dictionary_hi.json"
    model_dir: str = "models/indicxlit/ct2_int8"

    lru_cache_size: int = 10000
    log_level: str = "info"
    cors_origins: str = "*"             # comma-separated allowed origins

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Beam width caps candidate count; fail loudly if it cannot satisfy topk.
        validate_beam(self.beam_width, self.topk)


def get_settings() -> Settings:
    """Construct settings from the environment."""
    return Settings()
