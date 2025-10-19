# Test Configuration Files

This directory contains YAML configuration files used specifically for testing the `SequenceDataModule` and related components.

## Purpose

These configuration files are separate from the main project configurations to:
- Keep test configurations isolated from production configs
- Allow easy modification of test parameters without affecting main configs
- Provide clear examples of different configuration scenarios
- Enable consistent test behavior across different environments

## Configuration Files

### `test_basic_config.yaml`
Basic configuration with all dataset types (train/val/test) using fake datasets. Used for testing basic functionality without requiring real data.

### `test_real_data_config.yaml`
Configuration using the real LAMP dataset from Hugging Face. Used for integration tests with actual data.

### `test_streaming_config.yaml`
Configuration for testing streaming datasets. Includes streaming-specific parameters.

### `test_multiple_datasets_config.yaml`
Configuration with multiple training datasets to test dataset merging functionality.

## Usage in Tests

Tests load these configurations using the `load_test_config()` utility function:

```python
from tests.modelling.datamodules.test_seq_dm import load_test_config

# Load a specific test configuration
config = load_test_config("test_real_data_config")
```

## Important Notes

### Multiprocessing Settings
All test configurations use `num_workers: 0` in DataLoader settings to avoid multiprocessing warnings and potential deadlocks in test environments. This is a common practice for test configurations.

### Performance vs Testing
While `num_workers: 0` may be slower than using multiple workers, it's the recommended approach for tests because:
- Avoids multiprocessing warnings and potential deadlocks
- Makes tests more reliable and deterministic
- Reduces test execution complexity
- Prevents issues with fork() in multithreaded environments

## Adding New Test Configurations

When adding new test configurations:
1. Create a new YAML file with a descriptive name starting with `test_`
2. Follow the same structure as existing configs
3. Always use `num_workers: 0` for DataLoader configurations
4. Update relevant tests to use the new configuration
5. Document the purpose in this README
