# HyperPod CLI Documentation

This directory contains additional documentation for the SageMaker HyperPod CLI.

## Available Documentation

- [Slurm Cluster Management](slurm-cluster-management.md) - Full CRUD operations for Slurm-orchestrated HyperPod clusters

## Quick Links

### Slurm Cluster Commands

| Command | Description |
|---------|-------------|
| `hyp init slurm-cluster` | Initialize Slurm cluster configuration |
| `hyp create slurm-cluster` | Create a new Slurm cluster |
| `hyp list slurm-cluster` | List all Slurm clusters |
| `hyp describe slurm-cluster <name>` | Get cluster details |
| `hyp update slurm-cluster` | Update cluster configuration |
| `hyp delete slurm-cluster <name>` | Delete a cluster |
| `hyp update-software slurm-cluster <name>` | Apply AMI patches |
| `hyp list-nodes slurm-cluster <name>` | List cluster nodes |
| `hyp describe-node slurm-cluster <name>` | Get node details |
| `hyp scale-down slurm-cluster <name>` | Remove specific nodes |

For detailed usage, see the [Slurm Cluster Management](slurm-cluster-management.md) documentation.
