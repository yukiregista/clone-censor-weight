import sys
from contextlib import contextmanager
from logging import (
    DEBUG,
    INFO,
    WARNING,
    Formatter,
    Logger,
    LoggerAdapter,
    StreamHandler,
    _nameToLevel,
    getLogger,
)


class TagAdapter(LoggerAdapter):
    """Prefixes log messages with [tag]. Safe for libraries (no handlers)."""
    def __init__(self, logger: Logger, tag: str):
        super().__init__(logger, extra={"tag": tag})

    def process(self, msg, kwargs):
        tag = self.extra.get("tag", "")
        return f"[{tag}] {msg}", kwargs

def _has_effective_handlers(lg: Logger) -> bool:
    """Check whether this logger or its parents have any handlers."""
    cur: Logger | None = lg
    while cur:
        if cur.handlers:
            return True
        if not cur.propagate:
            break
        cur = cur.parent
    return False

LoggerLike = Logger | LoggerAdapter


def _unwrap_logger(lg: LoggerLike) -> Logger:
    # LoggerAdapter なら .logger を返す。通常の Logger はそのまま。
    return lg.logger if isinstance(lg, LoggerAdapter) else lg


@contextmanager
def temp_stderr_logging(logger: LoggerLike, level: int | None, *, user_supplied: bool = False):
    base = _unwrap_logger(logger)
    if level is None:
        yield; return

    old_level = base.level
    base.setLevel(level)
    handler = None
    try:
        # ユーザー提供ロガーならハンドラは増やさない、かつ何も設定されていない場合のみ一時的に stderr へ
        if not user_supplied and not _has_effective_handlers(base):
            handler = StreamHandler(sys.stderr)
            handler.setLevel(level)
            handler.setFormatter(Formatter("[%(levelname)s] %(name)s: %(message)s"))
            base.addHandler(handler)
        yield
    finally:
        if handler:
            base.removeHandler(handler)
        base.setLevel(old_level)

Verbosity = int | str | bool | None


def to_level(verbose: Verbosity) -> int | None:
    """
    Map a verbosity flag to a logging level.

    - None → None (don’t change anything)
    - False → WARNING
    - True → INFO
    - int:
        0 → WARNING
        1 → INFO
        2 or higher → DEBUG
    - str: "WARNING", "INFO", "DEBUG", etc.
    """
    if verbose is None:
        return None
    if isinstance(verbose, bool):
        return INFO if verbose else WARNING
    if isinstance(verbose, str):
        return _nameToLevel.get(verbose.upper(), INFO)
    if isinstance(verbose, int):
        if verbose <= 0:
            return WARNING
        elif verbose == 1:
            return INFO
        else:
            return DEBUG
    return None


def get_tagged(logger_like: LoggerLike | None, tag: str) -> TagAdapter:
    """
    Return a TagAdapter based on either a passed Logger/Adapter or this module's logger.
    If a parent adapter already has a tag, nest them as '[parent>tag]'.
    """
    base_logger = (
        logger_like.logger if isinstance(logger_like, LoggerAdapter)
        else logger_like
    ) or getLogger(__name__)

    # If caller passed an adapter with an existing tag, nest it.
    if isinstance(logger_like, LoggerAdapter):
        parent_tag = getattr(logger_like, "extra", {}).get("tag")
        if parent_tag:
            tag = f"{parent_tag}>{tag}"

    return TagAdapter(base_logger, tag)
