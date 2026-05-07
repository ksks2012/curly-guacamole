"""
Centralised logging configuration for the application.

Usage:
    from utils.logger import AppLogger

    # At application startup (dashboard / main entry point):
    AppLogger.setup(level="INFO", fmt=AppLogger.DEFAULT_FORMAT, datefmt=AppLogger.DEFAULT_DATEFMT)

    # In every module:
    log = AppLogger.get(__name__)

Constants can be overridden via etc/config.yaml:
    log_level:   INFO   # DEBUG | INFO | WARNING | ERROR
    log_format:  "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    log_datefmt: "%H:%M:%S"
"""

import logging


class AppLogger:
    """Application-wide logging setup and named-logger factory.

    Call AppLogger.setup() once at startup; then use AppLogger.get() in
    every module instead of logging.getLogger() directly. This ensures the
    format and level are always applied before any logger is first used.
    """

    DEFAULT_LEVEL = "INFO"
    DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    DEFAULT_DATEFMT = "%H:%M:%S"

    @classmethod
    def setup(
        cls,
        level: str = DEFAULT_LEVEL,
        fmt: str = DEFAULT_FORMAT,
        datefmt: str = DEFAULT_DATEFMT,
    ) -> None:
        """Configure the root logger. Must be called once at application startup.

        level   : log level name string, e.g. 'DEBUG', 'INFO', 'WARNING'
        fmt     : logging format string (Python logging format)
        datefmt : strftime-compatible date format for the asctime field
        """
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format=fmt,
            datefmt=datefmt,
        )

    @staticmethod
    def get(name: str) -> logging.Logger:
        """Return a named logger. Thin wrapper over logging.getLogger()."""
        return logging.getLogger(name)
