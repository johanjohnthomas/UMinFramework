"""
Configuration management system for UMinFramework.

This module provides flexible configuration management supporting both YAML and JSON
formats, with validation, environment variable overrides, and structured logging setup.
"""

import os
import json
import logging
import logging.config
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from copy import deepcopy
import warnings

# Try to import YAML support
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    warnings.warn("PyYAML not installed. YAML configuration support disabled.")


@dataclass
class ModelConfig:
    """Configuration for model loading and parameters."""
    name: str = "gpt2"
    device: Optional[str] = None
    torch_dtype: Optional[str] = None
    trust_remote_code: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    

@dataclass
class PromptRefinerConfig:
    """Configuration for prompt refinement."""
    enabled: bool = False
    model_path: Optional[str] = None
    max_length: int = 256
    num_beams: int = 4
    temperature: float = 0.7
    do_sample: bool = True


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty quantification."""
    enabled: bool = True
    method: str = "entropy"  # entropy, max_prob, margin, variance
    threshold: float = 0.7
    calibration_dataset: Optional[str] = None


@dataclass
class BacktrackingConfig:
    """Configuration for backtracking behavior."""
    enabled: bool = True
    window_size: int = 3
    max_backtracks_per_generation: int = 5
    max_backtracks_per_position: int = 2
    cot_templates: List[str] = field(default_factory=lambda: [
        " Let me think step by step.",
        " Let me reconsider this.",
        " Actually, let me think about this more carefully.",
        " Wait, let me approach this differently.",
        " Let me break this down:"
    ])


@dataclass
class GenerationConfig:
    """Configuration for text generation parameters."""
    max_length: int = 256
    min_length: int = 1
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_return_sequences: int = 1


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking and evaluation."""
    datasets: List[str] = field(default_factory=lambda: ["humaneval", "mbpp"])
    data_path: str = "data"
    output_dir: str = "results"
    max_problems: Optional[int] = None
    timeout: float = 30.0
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    qualitative_export: bool = True
    save_generated_code: bool = True
    export_qualitative_csv: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging behavior."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_logging: bool = True
    log_dir: str = "logs"
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5
    
    # Specific loggers
    token_level_logging: bool = False
    uncertainty_logging: bool = True
    backtrack_logging: bool = True


@dataclass
class UMinConfig:
    """Main configuration container for UMinFramework."""
    
    # Model configurations
    baseline_model: ModelConfig = field(default_factory=ModelConfig)
    augmented_model: ModelConfig = field(default_factory=ModelConfig)
    
    # Pipeline configurations
    prompt_refiner: PromptRefinerConfig = field(default_factory=PromptRefinerConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    backtracking: BacktrackingConfig = field(default_factory=BacktrackingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    # System configurations
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Metadata
    version: str = "1.0"
    description: str = "UMinFramework Configuration"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UMinConfig':
        """Create configuration from dictionary."""
        # Deep copy to avoid modifying original
        config_data = deepcopy(data)
        
        # Extract nested configurations
        baseline_model = ModelConfig(**config_data.pop('baseline_model', {}))
        augmented_model = ModelConfig(**config_data.pop('augmented_model', {}))
        prompt_refiner = PromptRefinerConfig(**config_data.pop('prompt_refiner', {}))
        uncertainty = UncertaintyConfig(**config_data.pop('uncertainty', {}))
        backtracking = BacktrackingConfig(**config_data.pop('backtracking', {}))
        generation = GenerationConfig(**config_data.pop('generation', {}))
        benchmark = BenchmarkConfig(**config_data.pop('benchmark', {}))
        logging_config = LoggingConfig(**config_data.pop('logging', {}))
        
        return cls(
            baseline_model=baseline_model,
            augmented_model=augmented_model,
            prompt_refiner=prompt_refiner,
            uncertainty=uncertainty,
            backtracking=backtracking,
            generation=generation,
            benchmark=benchmark,
            logging=logging_config,
            **config_data
        )


class ConfigManager:
    """Manager for loading, saving, and merging configurations."""
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory to look for configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        self.config_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Default configuration paths
        self.default_config_path = self.config_dir / "default.yaml"
        self.user_config_path = self.config_dir / "config.yaml"
        
        # Environment variable prefix
        self.env_prefix = "UMIN_"
    
    def create_default_config(self, path: Optional[Path] = None) -> UMinConfig:
        """Create and save a default configuration file."""
        if path is None:
            path = self.default_config_path
        
        default_config = UMinConfig()
        self.save_config(default_config, path)
        
        self.logger.info(f"Created default configuration at: {path}")
        return default_config
    
    def load_config(
        self, 
        config_path: Optional[Union[str, Path]] = None,
        create_if_missing: bool = True
    ) -> UMinConfig:
        """
        Load configuration from file with fallbacks.
        
        Args:
            config_path: Path to configuration file
            create_if_missing: Create default config if file doesn't exist
            
        Returns:
            UMinConfig instance
        """
        if config_path is None:
            config_path = self.user_config_path
        
        config_path = Path(config_path)
        
        # Check if file exists
        if not config_path.exists():
            if create_if_missing:
                self.logger.info(f"Configuration file not found at {config_path}")
                self.logger.info("Creating default configuration")
                return self.create_default_config(config_path)
            else:
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load configuration
        try:
            config_data = self._load_file(config_path)
            config = UMinConfig.from_dict(config_data)
            
            # Apply environment variable overrides
            config = self._apply_env_overrides(config)
            
            self.logger.info(f"Loaded configuration from: {config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_path}: {e}")
            if create_if_missing:
                self.logger.info("Using default configuration instead")
                return UMinConfig()
            raise
    
    def save_config(
        self, 
        config: UMinConfig, 
        config_path: Optional[Union[str, Path]] = None
    ):
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            config_path: Path where to save configuration
        """
        if config_path is None:
            config_path = self.user_config_path
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            config_data = config.to_dict()
            self._save_file(config_data, config_path)
            
            self.logger.info(f"Saved configuration to: {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration to {config_path}: {e}")
            raise
    
    def _load_file(self, file_path: Path) -> Dict[str, Any]:
        """Load data from YAML or JSON file."""
        suffix = file_path.suffix.lower()
        
        with open(file_path, 'r') as f:
            if suffix in ['.yaml', '.yml']:
                if not HAS_YAML:
                    raise ValueError("PyYAML not installed, cannot load YAML files")
                return yaml.safe_load(f)
            elif suffix == '.json':
                return json.load(f)
            else:
                # Try to detect format from content
                content = f.read()
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    if HAS_YAML:
                        return yaml.safe_load(content)
                    else:
                        raise ValueError(f"Cannot determine file format for: {file_path}")
    
    def _save_file(self, data: Dict[str, Any], file_path: Path):
        """Save data to YAML or JSON file."""
        suffix = file_path.suffix.lower()
        
        with open(file_path, 'w') as f:
            if suffix in ['.yaml', '.yml']:
                if not HAS_YAML:
                    raise ValueError("PyYAML not installed, cannot save YAML files")
                yaml.dump(data, f, default_flow_style=False, indent=2, sort_keys=True)
            elif suffix == '.json':
                json.dump(data, f, indent=2, sort_keys=True)
            else:
                # Default to JSON
                json.dump(data, f, indent=2, sort_keys=True)
    
    def _apply_env_overrides(self, config: UMinConfig) -> UMinConfig:
        """Apply environment variable overrides to configuration."""
        env_overrides = {}
        
        # Scan environment variables with our prefix
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(self.env_prefix):].lower()
                env_overrides[config_key] = value
        
        if env_overrides:
            self.logger.info(f"Found {len(env_overrides)} environment variable overrides")
            
            # Apply simple overrides (this could be expanded for nested configs)
            config_dict = config.to_dict()
            
            # Map common environment variables
            env_mapping = {
                'log_level': ['logging', 'level'],
                'uncertainty_threshold': ['uncertainty', 'threshold'],
                'backtrack_window': ['backtracking', 'window_size'],
                'max_length': ['generation', 'max_length'],
                'temperature': ['generation', 'temperature'],
                'data_path': ['benchmark', 'data_path'],
                'output_dir': ['benchmark', 'output_dir'],
            }
            
            for env_key, value in env_overrides.items():
                if env_key in env_mapping:
                    # Navigate to nested config
                    path = env_mapping[env_key]
                    current = config_dict
                    
                    for key in path[:-1]:
                        if key in current:
                            current = current[key]
                        else:
                            break
                    else:
                        # Try to convert value to appropriate type
                        final_key = path[-1]
                        if final_key in current:
                            try:
                                # Attempt type conversion based on existing value
                                existing_value = current[final_key]
                                if isinstance(existing_value, bool):
                                    current[final_key] = value.lower() in ['true', '1', 'yes', 'on']
                                elif isinstance(existing_value, int):
                                    current[final_key] = int(value)
                                elif isinstance(existing_value, float):
                                    current[final_key] = float(value)
                                else:
                                    current[final_key] = value
                                
                                self.logger.info(f"Applied env override: {env_key} = {value}")
                                
                            except (ValueError, TypeError) as e:
                                self.logger.warning(f"Failed to apply env override {env_key}: {e}")
            
            # Recreate config from modified dict
            config = UMinConfig.from_dict(config_dict)
        
        return config
    
    def merge_configs(self, base: UMinConfig, override: UMinConfig) -> UMinConfig:
        """Merge two configurations, with override taking precedence."""
        base_dict = base.to_dict()
        override_dict = override.to_dict()
        
        # Deep merge dictionaries
        merged_dict = self._deep_merge_dicts(base_dict, override_dict)
        
        return UMinConfig.from_dict(merged_dict)
    
    def _deep_merge_dicts(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result


class LoggingSetup:
    """Utility class for setting up structured logging based on configuration."""
    
    @staticmethod
    def setup_logging(config: LoggingConfig, log_dir: Optional[Path] = None):
        """
        Set up logging configuration.
        
        Args:
            config: Logging configuration
            log_dir: Directory for log files (overrides config.log_dir)
        """
        if log_dir is None:
            log_dir = Path(config.log_dir)
        
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logging configuration dictionary
        logging_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': config.format,
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
                'detailed': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': config.level,
                    'formatter': 'standard',
                    'stream': 'ext://sys.stdout'
                }
            },
            'loggers': {
                '': {  # Root logger
                    'level': config.level,
                    'handlers': ['console'],
                    'propagate': False
                }
            }
        }
        
        # Add file logging if enabled
        if config.file_logging:
            logging_config['handlers']['file'] = {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': config.level,
                'formatter': 'detailed',
                'filename': str(log_dir / 'umin_framework.log'),
                'maxBytes': config.max_bytes,
                'backupCount': config.backup_count,
                'encoding': 'utf8'
            }
            
            # Add file handler to root logger
            logging_config['loggers']['']['handlers'].append('file')
        
        # Set up specific loggers for different components
        component_loggers = {
            'umin_framework.uncertainty_head': {
                'level': 'DEBUG' if config.uncertainty_logging else 'INFO',
                'handlers': ['console'] + (['file'] if config.file_logging else []),
                'propagate': False
            },
            'umin_framework.generation_loop': {
                'level': 'DEBUG' if config.backtrack_logging else 'INFO',
                'handlers': ['console'] + (['file'] if config.file_logging else []),
                'propagate': False
            },
            'benchmark': {
                'level': 'INFO',
                'handlers': ['console'] + (['file'] if config.file_logging else []),
                'propagate': False
            }
        }
        
        logging_config['loggers'].update(component_loggers)
        
        # Apply configuration
        logging.config.dictConfig(logging_config)
        
        # Set up token-level logging if requested
        if config.token_level_logging:
            token_logger = logging.getLogger('umin_framework.tokens')
            token_logger.setLevel(logging.DEBUG)
            
            # Create separate handler for token logs
            token_handler = logging.handlers.RotatingFileHandler(
                log_dir / 'token_level.log',
                maxBytes=config.max_bytes,
                backupCount=config.backup_count
            )
            token_handler.setFormatter(logging.Formatter(
                '%(asctime)s - TOKEN - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            token_logger.addHandler(token_handler)
        
        logging.info("Logging configuration initialized")


# Global configuration instance
_global_config: Optional[UMinConfig] = None
_config_manager: Optional[ConfigManager] = None


def get_config() -> UMinConfig:
    """Get the global configuration instance."""
    global _global_config, _config_manager
    
    if _global_config is None:
        if _config_manager is None:
            _config_manager = ConfigManager()
        _global_config = _config_manager.load_config()
        
        # Set up logging
        LoggingSetup.setup_logging(_global_config.logging)
    
    return _global_config


def set_config(config: UMinConfig):
    """Set the global configuration instance."""
    global _global_config
    _global_config = config
    
    # Update logging configuration
    LoggingSetup.setup_logging(config.logging)


def load_config_from_file(config_path: Union[str, Path]) -> UMinConfig:
    """Load configuration from a specific file."""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager()
    
    config = _config_manager.load_config(config_path)
    set_config(config)
    
    return config


def save_current_config(config_path: Optional[Union[str, Path]] = None):
    """Save the current global configuration to file."""
    global _global_config, _config_manager
    
    if _global_config is None:
        raise ValueError("No configuration loaded")
    
    if _config_manager is None:
        _config_manager = ConfigManager()
    
    _config_manager.save_config(_global_config, config_path)


# Utility function for structured logging
def log_uncertainty_event(logger: logging.Logger, token: str, score: float, threshold: float):
    """Log uncertainty measurement event."""
    if score > threshold:
        logger.warning(f"High uncertainty detected - Token: '{token}', Score: {score:.3f} > {threshold:.3f}")
    else:
        logger.debug(f"Uncertainty score - Token: '{token}', Score: {score:.3f}")


def log_backtrack_event(logger: logging.Logger, position: int, window_size: int, reason: str):
    """Log backtracking event."""
    logger.warning(f"BACKTRACK - Position: {position}, Window: {window_size}, Reason: {reason}")


def log_generation_progress(logger: logging.Logger, tokens_generated: int, total_tokens: int):
    """Log generation progress."""
    progress = tokens_generated / total_tokens * 100 if total_tokens > 0 else 0
    logger.info(f"Generation progress: {tokens_generated}/{total_tokens} tokens ({progress:.1f}%)")


if __name__ == "__main__":
    # Demo usage
    print("UMinFramework Configuration System Demo")
    print("="*40)
    
    # Create config manager
    config_manager = ConfigManager(Path("demo_config"))
    
    # Create and save default configuration
    config = config_manager.create_default_config()
    
    print(f"Created default configuration")
    print(f"Uncertainty threshold: {config.uncertainty.threshold}")
    print(f"Backtracking enabled: {config.backtracking.enabled}")
    print(f"Log level: {config.logging.level}")
    
    # Demonstrate configuration loading
    loaded_config = config_manager.load_config()
    print(f"\nLoaded configuration from file")
    
    # Set up logging
    LoggingSetup.setup_logging(loaded_config.logging)
    
    # Demo structured logging
    logger = logging.getLogger('demo')
    logger.info("Configuration system initialized successfully")
    log_uncertainty_event(logger, "example", 0.8, 0.7)
    log_backtrack_event(logger, 42, 3, "high_uncertainty")
    
    print("Demo completed!")