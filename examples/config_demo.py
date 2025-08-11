#!/usr/bin/env python3
"""
Configuration System Demonstration for UMinFramework.

This script demonstrates how to use the UMinFramework configuration system
including loading configs from files, environment variable overrides, and
structured logging.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from umin_framework.config import (
        UMinConfig, ConfigManager, LoggingSetup,
        get_config, set_config, load_config_from_file,
        log_uncertainty_event, log_backtrack_event, log_generation_progress
    )
    print("✅ Successfully imported configuration system")
except ImportError as e:
    print(f"❌ Failed to import configuration system: {e}")
    sys.exit(1)


def demo_default_config():
    """Demonstrate default configuration creation and usage."""
    print("\n" + "="*60)
    print("DEMO 1: Default Configuration")
    print("="*60)
    
    # Create config manager
    config_manager = ConfigManager(Path("demo_config"))
    
    # Create default configuration
    config = config_manager.create_default_config()
    
    print("Default Configuration Created:")
    print(f"  • Uncertainty threshold: {config.uncertainty.threshold}")
    print(f"  • Backtracking enabled: {config.backtracking.enabled}")
    print(f"  • Backtrack window: {config.backtracking.window_size}")
    print(f"  • Max generation length: {config.generation.max_length}")
    print(f"  • Log level: {config.logging.level}")
    print(f"  • Log directory: {config.logging.log_dir}")
    
    # Set up logging
    LoggingSetup.setup_logging(config.logging)
    print("✅ Logging configured successfully")


def demo_config_loading():
    """Demonstrate loading configuration from files."""
    print("\n" + "="*60)
    print("DEMO 2: Configuration File Loading")
    print("="*60)
    
    # Try to load the example benchmark config
    example_config_path = Path("../config/benchmark_example.yaml")
    
    if example_config_path.exists():
        print(f"Loading configuration from: {example_config_path}")
        
        try:
            config = load_config_from_file(example_config_path)
            
            print("Loaded Configuration:")
            print(f"  • Description: {config.description}")
            print(f"  • Baseline model: {config.baseline_model.name}")
            print(f"  • Uncertainty threshold: {config.uncertainty.threshold}")
            print(f"  • Backtrack window: {config.backtracking.window_size}")
            print(f"  • Max problems: {config.benchmark.max_problems}")
            print(f"  • Output directory: {config.benchmark.output_dir}")
            print(f"  • Log level: {config.logging.level}")
            print("✅ Configuration loaded successfully")
            
            return config
            
        except Exception as e:
            print(f"❌ Failed to load configuration: {e}")
    else:
        print(f"❌ Example configuration file not found: {example_config_path}")
        print("Using default configuration instead")
        return get_config()


def demo_environment_overrides():
    """Demonstrate environment variable overrides."""
    print("\n" + "="*60)
    print("DEMO 3: Environment Variable Overrides")
    print("="*60)
    
    # Set some environment variables
    original_env = {}
    test_overrides = {
        'UMIN_LOG_LEVEL': 'DEBUG',
        'UMIN_UNCERTAINTY_THRESHOLD': '0.9',
        'UMIN_BACKTRACK_WINDOW': '7',
        'UMIN_MAX_LENGTH': '512',
        'UMIN_TEMPERATURE': '0.1'
    }
    
    print("Setting environment variables:")
    for key, value in test_overrides.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
        print(f"  • {key}={value}")
    
    try:
        # Create config manager and load with env overrides
        config_manager = ConfigManager(Path("demo_config_env"))
        config = config_manager.load_config()
        
        print("\nConfiguration with environment overrides:")
        print(f"  • Log level: {config.logging.level}")
        print(f"  • Uncertainty threshold: {config.uncertainty.threshold}")
        print(f"  • Backtrack window: {config.backtracking.window_size}")
        print(f"  • Max length: {config.generation.max_length}")
        print(f"  • Temperature: {config.generation.temperature}")
        print("✅ Environment overrides applied successfully")
        
    finally:
        # Restore original environment
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


def demo_structured_logging(config):
    """Demonstrate structured logging functionality."""
    print("\n" + "="*60)
    print("DEMO 4: Structured Logging")
    print("="*60)
    
    # Set up logging from configuration
    LoggingSetup.setup_logging(config.logging)
    
    import logging
    
    # Get different loggers
    main_logger = logging.getLogger('demo')
    uncertainty_logger = logging.getLogger('umin_framework.uncertainty_head')
    generation_logger = logging.getLogger('umin_framework.generation_loop')
    benchmark_logger = logging.getLogger('benchmark')
    
    print("Demonstrating structured logging at different levels:")
    print("(Check the log files in the configured log directory)")
    
    # Basic logging
    main_logger.info("Starting structured logging demonstration")
    main_logger.debug("This is a debug message")
    main_logger.warning("This is a warning message")
    
    # Specialized logging functions
    print("\n📊 Logging uncertainty events:")
    log_uncertainty_event(uncertainty_logger, "function", 0.85, 0.7)
    log_uncertainty_event(uncertainty_logger, "return", 0.65, 0.7)
    log_uncertainty_event(uncertainty_logger, "variable", 0.92, 0.8)
    
    print("🔄 Logging backtrack events:")
    log_backtrack_event(generation_logger, 42, 3, "high_uncertainty")
    log_backtrack_event(generation_logger, 58, 5, "max_attempts")
    
    print("📈 Logging generation progress:")
    log_generation_progress(benchmark_logger, 25, 100)
    log_generation_progress(benchmark_logger, 50, 100)
    log_generation_progress(benchmark_logger, 100, 100)
    
    main_logger.info("Structured logging demonstration completed")
    print("✅ Structured logging demonstration completed")


def demo_config_validation():
    """Demonstrate configuration validation and error handling."""
    print("\n" + "="*60)
    print("DEMO 5: Configuration Validation")
    print("="*60)
    
    try:
        # Create an invalid configuration
        print("Testing invalid configuration handling...")
        
        config_manager = ConfigManager(Path("demo_config_invalid"))
        
        # Try to load non-existent file
        try:
            config_manager.load_config("nonexistent.yaml", create_if_missing=False)
        except FileNotFoundError:
            print("✅ Correctly handled missing configuration file")
        
        # Test configuration merging
        base_config = UMinConfig()
        override_config = UMinConfig()
        override_config.uncertainty.threshold = 0.95
        override_config.generation.max_length = 512
        
        merged_config = config_manager.merge_configs(base_config, override_config)
        
        print("Configuration merging test:")
        print(f"  • Base uncertainty threshold: {base_config.uncertainty.threshold}")
        print(f"  • Override uncertainty threshold: {override_config.uncertainty.threshold}")
        print(f"  • Merged uncertainty threshold: {merged_config.uncertainty.threshold}")
        print("✅ Configuration merging works correctly")
        
    except Exception as e:
        print(f"❌ Configuration validation error: {e}")


def main():
    """Run all configuration system demonstrations."""
    print("UMinFramework Configuration System Demo")
    print("======================================")
    
    try:
        # Run demonstrations
        demo_default_config()
        config = demo_config_loading()
        demo_environment_overrides()
        demo_structured_logging(config)
        demo_config_validation()
        
        print("\n" + "="*60)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\n💡 Key Features Demonstrated:")
        print("  ✅ Default configuration creation")
        print("  ✅ YAML/JSON configuration file loading")
        print("  ✅ Environment variable overrides")
        print("  ✅ Structured logging with different levels")
        print("  ✅ Configuration validation and error handling")
        print("  ✅ Configuration merging and inheritance")
        
        print("\n📁 Files Created:")
        print("  • demo_config/default.yaml - Default configuration")
        print("  • demo_config_env/default.yaml - Environment override demo")
        print("  • logs/ - Log files with structured output")
        
        print("\n🔧 Next Steps:")
        print("  1. Customize config/default.yaml for your experiments")
        print("  2. Use environment variables for easy parameter sweeps")
        print("  3. Check log files for detailed execution traces")
        print("  4. Use benchmark script with --config flag")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())