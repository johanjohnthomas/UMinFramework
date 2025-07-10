"""
Configuration module for UMinFramework.

This module contains configuration classes and utilities for managing
uncertainty minimization settings.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import json
import os


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty detection and minimization."""
    
    # Model settings
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    uhead_name: str = "llm-uncertainty-head/uhead_Mistral-7B-Instruct-v0.2"
    device: str = "cuda"
    
    # Uncertainty detection settings
    uncertainty_threshold: float = 0.5
    uncertainty_method: str = "softmax"  # "softmax", "entropy", "variance"
    
    # Backtracking settings
    max_backtrack_attempts: int = 3
    backtrack_strategy: str = "first_uncertain"  # "first_uncertain", "highest_uncertain", "sliding_window"
    
    # Chain-of-Thought settings
    cot_trigger_token: str = "<think>"
    cot_strategy: str = "simple"  # "simple", "structured", "progressive"
    
    # Generation settings
    max_length: int = 512
    temperature: float = 0.7
    do_sample: bool = True
    top_p: float = 0.9
    top_k: int = 50
    
    # Logging and monitoring
    log_level: str = "INFO"
    save_generation_history: bool = True
    history_file: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'UncertaintyConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'UncertaintyConfig':
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def save_to_json(self, json_path: str):
        """Save configuration to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Predefined configurations for different use cases
CONFIGS = {
    "conservative": UncertaintyConfig(
        uncertainty_threshold=0.3,
        max_backtrack_attempts=5,
        temperature=0.5,
        cot_strategy="structured"
    ),
    
    "balanced": UncertaintyConfig(
        uncertainty_threshold=0.5,
        max_backtrack_attempts=3,
        temperature=0.7,
        cot_strategy="simple"
    ),
    
    "aggressive": UncertaintyConfig(
        uncertainty_threshold=0.7,
        max_backtrack_attempts=2,
        temperature=0.9,
        cot_strategy="progressive"
    ),
    
    "research": UncertaintyConfig(
        uncertainty_threshold=0.4,
        max_backtrack_attempts=4,
        temperature=0.6,
        save_generation_history=True,
        log_level="DEBUG"
    )
}


def get_config(config_name: str = "balanced") -> UncertaintyConfig:
    """
    Get a predefined configuration.
    
    Args:
        config_name: Name of the configuration to retrieve
        
    Returns:
        UncertaintyConfig instance
    """
    if config_name not in CONFIGS:
        raise ValueError(f"Configuration '{config_name}' not found. Available: {list(CONFIGS.keys())}")
    
    return CONFIGS[config_name]


def create_custom_config(**kwargs) -> UncertaintyConfig:
    """
    Create a custom configuration by overriding default values.
    
    Args:
        **kwargs: Configuration parameters to override
        
    Returns:
        UncertaintyConfig instance with custom settings
    """
    base_config = UncertaintyConfig()
    
    # Update with custom values
    for key, value in kwargs.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
    
    return base_config
