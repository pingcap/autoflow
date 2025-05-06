import logging
from logging.config import dictConfig
from app.core.config import settings

logger = logging.getLogger("api_server")


dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            },
        },
        "root": {
            "level": settings.LOG_LEVEL,
            "handlers": ["console"],
        },
        "loggers": {
            "uvicorn.error": {
                "level": "ERROR",
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "sqlalchemy.engine": {
                "level": settings.SQLALCHEMY_LOG_LEVEL,
                "handlers": ["console"],
                "propagate": False,
            },
        },
    }
)
