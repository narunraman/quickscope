"""
Centralized logging configuration for the Quickscope framework.
Provides structured logging with different levels and categories.

Adds file logging to a centralized logs directory at the project root by default.
"""

import logging
from pathlib import Path
import sys
from typing import Optional


# Define custom TRACE level
TRACE = 5
logging.addLevelName(TRACE, 'TRACE')


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to log levels."""
    
    COLORS = {
        'TRACE': '\033[90m',      # Dark gray
        'DEBUG': '\033[94m',      # Blue  
        'INFO': '\033[92m',       # Green
        'WARNING': '\033[93m',    # Yellow
        'ERROR': '\033[91m',      # Red
        'CRITICAL': '\033[95m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to the log level
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        
        return super().format(record)


def setup_logging(
    level: int = logging.INFO,
    use_colors: bool = True,
    log_format: Optional[str] = None,
    *,
    log_to_file: bool = True,
    log_dir: Optional[str] = None,
    log_filename: Optional[str] = None,
) -> None:
    """
    Setup centralized logging for Quickscope.
    
    Args:
        level: Base logging level (default: INFO)
        use_colors: Whether to use colored output (default: True)
        log_format: Custom log format string
        log_to_file: Whether to also log to a file under a central directory (default: True)
        log_dir: Directory for log files (default: '<project_root>/logs')
        log_filename: Log file name (default: 'quickscope.log')
    """
    if log_format is None:
        log_format = '%(levelname)s - %(message)s'
    
    # Create formatter
    if use_colors and sys.stdout.isatty():
        formatter = ColoredFormatter(log_format, datefmt='%H:%M:%S')
    else:
        formatter = logging.Formatter(log_format, datefmt='%H:%M:%S')
    
    # Configure root logger
    root_logger = logging.getLogger('quickscope')
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    # Stream handler (console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # Optional file handler
    if log_to_file:
        # Resolve project root (two levels up from this file: <root>/quickscope/logging_config.py)
        project_root = Path(__file__).resolve().parent.parent
        resolved_dir = Path(log_dir) if log_dir else project_root / 'logs'
        resolved_dir.mkdir(parents=True, exist_ok=True)

        filename = log_filename or 'quickscope.log'
        file_path = resolved_dir / filename

        file_formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s - %(message)s',
                                           datefmt='%Y-%m-%d %H:%M:%S')
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
    
    # Prevent duplicate logging
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the specified component.
    
    Args:
        name: Logger name (e.g., 'progress', 'api', 'parsing')
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(f'quickscope.{name}')


# Pre-configured loggers for different components
progress_logger = get_logger('progress')
api_logger = get_logger('api') 
parsing_logger = get_logger('parsing')
response_logger = get_logger('responses')
config_logger = get_logger('config')


def set_level_by_name(level_name: str) -> None:
    """
    Set logging level by name.
    
    Args:
        level_name: One of 'TRACE', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    """
    level_map = {
        'TRACE': TRACE,
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO, 
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    level = level_map.get(level_name.upper(), logging.INFO)
    logging.getLogger('quickscope').setLevel(level)


def set_component_level(component: str, level: int) -> None:
    """
    Set logging level for a specific component.
    
    Args:
        component: Component name ('progress', 'api', 'parsing', 'responses', 'config')
        level: Logging level
    """
    logger = get_logger(component)
    logger.setLevel(level)


# Convenience functions for common logging patterns
def log_progress(message: str) -> None:
    """Log progress information."""
    progress_logger.info(message)


def log_config(message: str) -> None:
    """Log configuration information.""" 
    config_logger.info(message)


def log_api_call(message: str, level: int = logging.DEBUG) -> None:
    """Log API call information."""
    api_logger.log(level, message)


def log_parsing(message: str, level: int = logging.DEBUG) -> None:
    """Log parsing information."""
    parsing_logger.log(level, message)


def log_response(message: str, level: int = TRACE) -> None:
    """Log full response content."""
    response_logger.log(level, message)
