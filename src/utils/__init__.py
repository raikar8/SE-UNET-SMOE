"""
Utility functions and classes
Configuration, device management, logging, and helper functions

"""

"""
Utility functions and classes
Configuration, device management, logging, and helper functions
"""

from .config import (
    ExperimentConfig,
    ModelConfig,
    AudioConfig,
    DeviceConfig,
    SMoEConfig,
    TrainingConfig,
    DataConfig,
    ModelType,
    LossType,
    ActivationType,
    create_mac_optimized_config
)

from .device_utils import (
    DeviceType,
    DeviceInfo,
    detect_best_device,
    setup_device_optimizations,
    move_to_device,
    get_memory_usage,
    clear_cache
)

from .logging_utils import (
    setup_logging,
    get_logger,
    log_model_info,
    log_training_progress
)

__all__ = [
    # Configuration classes
    "ExperimentConfig",
    "ModelConfig",
    "AudioConfig",
    "DeviceConfig",
    "SMoEConfig",
    "TrainingConfig",
    "DataConfig",

    # Configuration enums
    "ModelType",
    "LossType",
    "ActivationType",

    # Configuration functions
    "create_mac_optimized_config",

    # Device management
    "DeviceType",
    "DeviceInfo",
    "detect_best_device",
    "setup_device_optimizations",
    "move_to_device",
    "get_memory_usage",
    "clear_cache",

    # Logging
    "setup_logging",
    "get_logger",
    "log_model_info",
    "log_training_progress"
]
