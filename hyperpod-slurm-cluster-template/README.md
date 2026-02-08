# HyperPod Slurm Cluster Template

Versioned JSON-schema and Pydantic models for SageMaker HyperPod Slurm cluster configuration.

## Overview

This package provides configuration templates and validation models for creating Slurm-orchestrated SageMaker HyperPod clusters via the HyperPod CLI.

## Usage

This package is used internally by the HyperPod CLI when running:

```bash
hyp init slurm-cluster
```

This command scaffolds the configuration files needed to create a Slurm cluster:
- `config.yaml` - Main cluster configuration
- `provisioning_parameters.json` - Slurm-specific provisioning parameters

## Version History

- **1.0** - Initial release with support for Slurm cluster configuration
