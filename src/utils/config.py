"""
Configuration Management
Centralized configuration using dataclasses and validation
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import yaml
import json
from enum import Enum

class ModelType(Enum):
    """Supported model architectures"""
    DCUNET_10 = "dcunet10"
    DCUNET_20 = "dcunet20"
    SMOE_DCUNET_10 = "smoe_dcunet10"
    SMOE_DCUNET_20 = "smoe_dcunet20"

class LossType(Enum):
    """Supported loss functions"""
    WEIGHTED_SDR = "weighted_sdr"
    MSE = "mse"
    L1 = "l1"
    SPECTRAL_MSE = "spectral_mse"

class ActivationType(Enum):
    """Supported activation functions"""
    CRELU = "crelu"
    ZRELU = "zrelu"
    MODRELU = "modrelu"

@dataclass
class AudioConfig:
    """Audio processing configuration"""

    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 256
    win_length: Optional[int] = None
    window_type: str ='hann'
    segment_length: int = 16000

    def __post_init__(self):
        if self.win_length is None:
            self.win_length = self.n_fft

        if self.hop_length > self.n_fft:
            raise ValueError(f"hop_length ({self.hop_length}) must be <= n_fft ({self.n_fft})")

        if self.segment_length < self.n_fft:
            raise ValueError(f"segment_length ({self.segment_length}) must be >= n_fft ({self.n_fft})")

@dataclass
class ModelConfig:
    """Base model architecture configuration"""

    # Architecture parameters
    depth: int = 20
    init_channels: int = 45
    growth_factor: float = 2.0

    # Layer Configuration
    use_batch_norm: bool = True
    activation : str = ActivationType.CRELU.value
    dropout: float = 0.0

    # Loss Configuration
    loss_type : str = LossType.WEIGHTED_SDR.value

    def __post_init__(self):
        if self.depth not in [10, 20]:
            raise ValueError(f"Unsupported depth: ({self.depth}), must be either 10 or 20")

        if self.init_channels <= 0:
            raise ValueError(f"Unsupported init_channels: ({self.init_channels}), must be > 0")

        if self.growth_factor <= 1:
            raise ValueError(f"Unsupported growth_factor: ({self.growth_factor}), must be > 1")

        if not (0.0 <= self.growth_factor <= 1.0):
            raise ValueError(f"Unsupported growth_factor: ({self.growth_factor}), must be >= 0 and <= 1")

        # Validate enum values

        try:
            ActivationType(self.activation)
        except ValueError:
            valid_activations = [e.value for e in ActivationType]
            raise ValueError(f"Invalid activation: {self.activation}. Valid: {valid_activations}")

        try:
            LossType(self.loss_type)
        except ValueError:
            valid_loss = [e.value for e in LossType]
            raise ValueError(f"Invalid loss_type: {self.loss_type}. Valid: {valid_loss}")

    @property
    def n_layers(self) -> int:
        """Number of layers in the model based on Depth"""
        if self.depth == 10:
            return 4
        elif self.depth == 20:
            return 5
        else:
            return max(3, self.depth // 4)

    def channel_progression(self) -> List[int]:
        """Channel progression through the Network"""
        channels = [self.init_channels]
        for i in range (self.n_layers):
            next_ch = int(channels[-1] * self.growth_factor)
            channels.append(next_ch)

        return channels

@dataclass
class SMoEConfig:
    """SMOE  architecture configuration"""

    # Expert Structure
    n_experts : int = 2
    expert_hidden_multiplier: float = 2.0
    expert_dropout: float = 0.1

    # Routing Configuration
    use_implicit_routing: bool = True
    routing_temperature: float = 1.0
    confidence_threshold: float = 0.7

    # Multi-expert activation
    enable_multi_experts: bool = True
    max_active_experts: int = 2

    # Expert Placement
    smoe_layer_positions: List[str] = field(default_factory=lambda: [
        'early_encoder', 'mid_encoder', 'bottleneck', 'mid_decoder', 'late_decoder'
    ])

    # Training parameters
    router_lr_multiplier: float = 2.0

    def __post_init__(self):
        """Validate S-MoE configuration"""
        if self.n_experts < 2:
            raise ValueError(f"n_experts must be >= 2, got {self.n_experts}")

        if self.expert_hidden_multiplier <= 0:
            raise ValueError(f"expert_hidden_multiplier must be > 0, got {self.expert_hidden_multiplier}")

        if not (0.0 <= self.expert_dropout <= 1.0):
            raise ValueError(f"expert_dropout must be in [0, 1], got {self.expert_dropout}")

        if self.routing_temperature <= 0:
            raise ValueError(f"routing_temperature must be > 0, got {self.routing_temperature}")

        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}")

        if self.max_active_experts > self.n_experts:
            raise ValueError(
                f"max_active_experts ({self.max_active_experts}) cannot exceed n_experts ({self.n_experts})")

        # Validate layer positions
        valid_positions = {'early_encoder', 'mid_encoder', 'bottleneck', 'mid_decoder', 'late_decoder'}
        invalid_positions = set(self.smoe_layer_positions) - valid_positions
        if invalid_positions:
            raise ValueError(f"Invalid layer positions: {invalid_positions}. Valid: {valid_positions}")


@dataclass
class TrainingConfig:
    """Training configuration"""

    # Basic training parameters
    num_epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # Learning rate scheduling
    scheduler_type: str = 'reduce_on_plateau'
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5

    # Optimization
    optimizer: str = 'adamw'
    gradient_clip_norm: float = 1.0
    use_amp: bool = True  # Automatic Mixed Precision

    # Validation and checkpointing
    validate_every: int = 1  # Validate every N epochs
    save_every: int = 5  # Save checkpoint every N epochs
    save_best_only: bool = False

    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4

    def __post_init__(self):
        """Validate training configuration"""
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")

        if self.gradient_clip_norm <= 0:
            raise ValueError(f"gradient_clip_norm must be positive, got {self.gradient_clip_norm}")

        valid_optimizers = {'adam', 'adamw', 'sgd', 'rmsprop'}
        if self.optimizer.lower() not in valid_optimizers:
            raise ValueError(f"Invalid optimizer: {self.optimizer}. Valid: {valid_optimizers}")

        valid_schedulers = {'reduce_on_plateau', 'cosine', 'step', 'exponential', 'none'}
        if self.scheduler_type.lower() not in valid_schedulers:
            raise ValueError(f"Invalid scheduler: {self.scheduler_type}. Valid: {valid_schedulers}")


@dataclass
class DataConfig:
    """Data loading and processing configuration"""

    # Dataset paths
    clean_train_dir: Optional[str] = None
    noisy_train_dir: Optional[str] = None
    clean_val_dir: Optional[str] = None
    noisy_val_dir: Optional[str] = None
    distortion_labels_file: Optional[str] = None

    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    # Data splitting
    train_val_split: float = 0.9
    random_seed: int = 42

    # Data augmentation
    enable_augmentation: bool = True
    augmentation_params: Dict[str, Any] = field(default_factory=lambda: {
        'gain_range': (0.7, 1.3),
        'snr_range': (0.5, 2.0),
        'gain_prob': 0.5,
        'snr_prob': 0.3
    })

    def __post_init__(self):
        """Validate data configuration"""
        if not (0.0 < self.train_val_split < 1.0):
            raise ValueError(f"train_val_split must be in (0, 1), got {self.train_val_split}")

        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")


@dataclass
class ExperimentConfig:
    """Complete Experiment configuration"""

    # Sub-configurations
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    smoe: Optional[SMoEConfig] = None

    # Experiment metadata
    experiment_name: str = "speech_enhancement_experiment"
    model_type: str = ModelType.DCUNET_20.value
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Output paths
    output_dir: str = "./outputs"
    checkpoint_dir: Optional[str] = None
    log_dir: Optional[str] = None

    # Logging and monitoring
    log_level: str = "INFO"
    use_wandb: bool = False
    wandb_project: str = "speech-enhancement"
    wandb_entity: Optional[str] = None

    # Reproducibility
    random_seed: int = 42
    deterministic: bool = False

    def __post_init__(self):
        """Set up derived paths and validate configuration"""
        # Set up output directories
        output_path = Path(self.output_dir)

        if self.checkpoint_dir is None:
            self.checkpoint_dir = str(output_path / "checkpoints")

        if self.log_dir is None:
            self.log_dir = str(output_path / "logs")

        # Validate model type
        try:
            ModelType(self.model_type)
        except ValueError:
            valid_types = [e.value for e in ModelType]
            raise ValueError(f"Invalid model_type: {self.model_type}. Valid: {valid_types}")

        # Validate S-MoE configuration requirement
        if self.model_type.startswith('smoe') or self.model_type == 'hierarchical_moe':
            if self.smoe is None:
                self.smoe = SMoEConfig()

        # Validate log level
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"Invalid log_level: {self.log_level}. Valid: {valid_levels}")


    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'ExperimentConfig':
        """Load experiment configuration from YAML file"""

        config_path = Path(config_path)

        if not config_path.exists():
            raise ValueError(f"Config file {config_path} does not exist")

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, config_path: Union[str, Any]) -> 'ExperimentConfig':
        """Load Configuration from JSON file"""

        config_path = Path(config_path)
        if not config_path.exists():
            raise ValueError(f"Config file {config_path} does not exist")

        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':

        """Create configuration from dictionary"""

        # Extract sub-configuration dictionaries

        audio_dict = config_dict.get('audio',{})
        model_dict = config_dict.get('model',{})
        training_dict = config_dict.get('training',{})
        data_dict = config_dict.get('data',{})
        smoe_dict = config_dict.get('smoe',None)

        # Create sub-configurations

        audio_config = AudioConfig(**audio_dict)
        model_config = ModelConfig(**model_dict)
        training_config = TrainingConfig(**training_dict)
        data_config = DataConfig(**data_dict)
        smoe_config = SMoEConfig(**smoe_dict) if smoe_dict else None

        # Create main configuration
        main_dict = {k: v for k, v in config_dict.items()
                     if k not in ['audio', 'model', 'training', 'data', 'smoe']}

        return cls(
            audio=audio_config,
            model=model_config,
            training=training_config,
            data=data_config,
            smoe=smoe_config,
            **main_dict
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'audio': self.audio.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'smoe': self.smoe.__dict__ if self.smoe else None,
            'experiment_name': self.experiment_name,
            'model_type': self.model_type,
            'description': self.description,
            'tags': self.tags,
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir,
            'log_level': self.log_level,
            'use_wandb': self.use_wandb,
            'wandb_project': self.wandb_project,
            'wandb_entity': self.wandb_entity,
            'random_seed': self.random_seed,
            'deterministic': self.deterministic
        }

    def save_yaml(self, save_path: Union[str, Path]) -> None:
        """Save configuration to YAML file"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

    def save_json(self, save_path: Union[str, Path]) -> None:
        """Save configuration to JSON file"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def create_directories(self) -> None:
        """Create all necessary output directories"""
        directories = [
            self.output_dir,
            self.checkpoint_dir,
            self.log_dir
        ]

        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)


def create_default_configs() -> Dict[str, ExperimentConfig]:
    """Create default configurations for different model types"""
    configs = {}

    # DCUNet-20 baseline
    configs['dcunet20'] = ExperimentConfig(
        experiment_name="dcunet20_baseline",
        model_type=ModelType.DCUNET_20.value,
        description="Baseline Deep Complex U-Net with 20 layers"
    )

    # S-MoE DCUNet-20
    configs['smoe_dcunet20'] = ExperimentConfig(
        experiment_name="smoe_dcunet20",
        model_type=ModelType.SMOE_DCUNET_20.value,
        description="S-MoE enhanced Deep Complex U-Net",
        smoe=SMoEConfig(
            n_experts=2,
            use_implicit_routing=True,
            enable_multi_experts=True
        )
    )

    # Two-expert system
    configs['two_expert'] = ExperimentConfig(
        experiment_name="two_expert_dcunet20",
        model_type=ModelType.SMOE_DCUNET_20.value,
        description="2-Expert system for noise and reverberation",
        smoe=SMoEConfig(
            n_experts=2,
            expert_hidden_multiplier=1.5,
            confidence_threshold=0.6,
            smoe_layer_positions=['mid_encoder', 'bottleneck', 'mid_decoder']
        )
    )

    return configs


def test_config():
    """Test configuration loading and validation"""

    # Test default configuration
    config = ExperimentConfig()
    print("✅ Default configuration created successfully")

    # Test configuration validation
    try:
        invalid_config = ExperimentConfig(
            model=ModelConfig(depth=15)  # Invalid depth
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✅ Configuration validation works: {e}")

    # Test dictionary conversion
    config_dict = config.to_dict()
    restored_config = ExperimentConfig.from_dict(config_dict)
    print("✅ Dictionary conversion works")

    # Test default configs
    default_configs = create_default_configs()
    assert len(default_configs) == 3, f"Expected 3 default configs, got {len(default_configs)}"
    print("✅ Default configurations created successfully")

    print("✅ All configuration tests passed!")


if __name__ == "__main__":
    test_config()
























