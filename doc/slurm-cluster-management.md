# Slurm Cluster Management

This document describes the CLI commands for managing Slurm-orchestrated SageMaker HyperPod clusters.

## Overview

The HyperPod CLI now supports full CRUD operations for Slurm-orchestrated clusters, providing feature parity with EKS cluster management. Unlike EKS clusters which use CloudFormation stacks, Slurm clusters interact directly with the SageMaker API.

## Commands

### Initialize Slurm Cluster Configuration

Scaffold configuration files for a new Slurm cluster:

```bash
hyp init slurm-cluster
```

This creates:
- `config.yaml` - Main configuration file with cluster settings
- `provisioning_parameters.json` - Slurm-specific settings (controller group, worker groups, FSx mount)

You can also specify a target directory:

```bash
hyp init slurm-cluster <directory>
```

### Validate Configuration

Validate the configuration before creating a cluster:

```bash
hyp validate
```

This checks:
- YAML schema compliance
- IAM role ARN format
- S3 URI format
- Required field presence

### Create Slurm Cluster

Create a cluster from the configuration:

```bash
hyp create slurm-cluster
```

| Option | Type | Description |
|--------|------|-------------|
| `--region <region>` | Optional | AWS region for the cluster |
| `--wait` | Optional | Wait for cluster to reach InService or Failed state |
| `--debug` | Optional | Enable debug mode |

### List Slurm Clusters

List all Slurm-orchestrated clusters:

```bash
hyp list slurm-cluster
```

| Option | Type | Description |
|--------|------|-------------|
| `--region <region>` | Optional | AWS region to list clusters from |
| `--output <json\|table>` | Optional | Output format (default: json) |
| `--debug` | Optional | Enable debug mode |

### Describe Slurm Cluster

Get detailed information about a cluster:

```bash
hyp describe slurm-cluster <cluster-name>
```

| Option | Type | Description |
|--------|------|-------------|
| `--region <region>` | Optional | AWS region where the cluster exists |
| `--output <json\|table>` | Optional | Output format (default: json) |
| `--debug` | Optional | Enable debug mode |

### Update Slurm Cluster

Update cluster configuration (instance groups, node recovery):

```bash
hyp update slurm-cluster --cluster-name <name> \
    --instance-groups '[{"InstanceCount":4,"InstanceGroupName":"worker-group","InstanceType":"ml.p4d.24xlarge"}]' \
    --node-recovery Automatic
```

| Option | Type | Description |
|--------|------|-------------|
| `--cluster-name <name>` | Required | Name of the cluster to update |
| `--instance-groups <json>` | Optional | New instance group configuration |
| `--node-recovery <value>` | Optional | Node recovery setting (Automatic/None) |
| `--region <region>` | Optional | AWS region |
| `--debug` | Optional | Enable debug mode |

### Delete Slurm Cluster

Delete a Slurm cluster:

```bash
hyp delete slurm-cluster <cluster-name> --region <region>
```

| Option | Type | Description |
|--------|------|-------------|
| `--region <region>` | Required | AWS region where the cluster exists |
| `--debug` | Optional | Enable debug mode |

### Update Cluster Software

Apply AMI patches to cluster nodes:

```bash
hyp update-software slurm-cluster <cluster-name>
```

| Option | Type | Description |
|--------|------|-------------|
| `--region <region>` | Optional | AWS region |
| `--debug` | Optional | Enable debug mode |

### List Cluster Nodes

List all nodes in a cluster:

```bash
hyp list-nodes slurm-cluster <cluster-name>
```

| Option | Type | Description |
|--------|------|-------------|
| `--region <region>` | Optional | AWS region |
| `--output <json\|table>` | Optional | Output format (default: json) |
| `--debug` | Optional | Enable debug mode |

### Describe Cluster Node

Get detailed information about a specific node:

```bash
hyp describe-node slurm-cluster <cluster-name> --node-id <node-id>
```

| Option | Type | Description |
|--------|------|-------------|
| `--node-id <id>` | Required | Node ID to describe |
| `--region <region>` | Optional | AWS region |
| `--output <json\|table>` | Optional | Output format (default: json) |
| `--debug` | Optional | Enable debug mode |

### Scale Down Cluster

Remove specific nodes from a cluster:

```bash
hyp scale-down slurm-cluster <cluster-name> --node-ids <id1,id2,id3>
```

| Option | Type | Description |
|--------|------|-------------|
| `--node-ids <ids>` | Required | Comma-separated list of node IDs to remove |
| `--region <region>` | Optional | AWS region |
| `--debug` | Optional | Enable debug mode |

## Configuration File Format

### config.yaml

```yaml
template: slurm-cluster
version: "1.0"

cluster_name: my-slurm-cluster

instance_groups:
  - instance_group_name: controller-group
    instance_type: ml.t3.medium
    instance_count: 1
    life_cycle_config:
      source_s3_uri: s3://my-bucket/lifecycle-scripts/
      on_create: on_create.sh
    execution_role: arn:aws:iam::123456789012:role/HyperPodRole
    threads_per_core: 1
    instance_storage_configs:
      - ebs_volume_config:
          volume_size_in_gb: 100

  - instance_group_name: worker-group
    instance_type: ml.p4d.24xlarge
    instance_count: 4
    life_cycle_config:
      source_s3_uri: s3://my-bucket/lifecycle-scripts/
      on_create: on_create.sh
    execution_role: arn:aws:iam::123456789012:role/HyperPodRole
    threads_per_core: 2

vpc_config:
  security_group_ids:
    - sg-0123456789abcdef0
  subnets:
    - subnet-0123456789abcdef0

node_recovery: Automatic

tags:
  - key: Environment
    value: Production

orchestrator:
  eks: null  # Must be null for Slurm clusters
```

### provisioning_parameters.json

```json
{
  "version": "1.0",
  "workload_manager": "slurm",
  "controller_group": "controller-group",
  "worker_groups": [
    {
      "instance_group_name": "worker-group",
      "partition_name": "ml-workers"
    }
  ],
  "fsx_dns_name": "fs-0123456789abcdef0.fsx.us-west-2.amazonaws.com",
  "fsx_mountname": "/fsx"
}
```

## Workflow Example

A typical workflow for creating a Slurm cluster:

1. Initialize configuration:
   ```bash
   hyp init slurm-cluster
   ```

2. Edit `config.yaml` with your settings (cluster name, instance types, VPC config, etc.)

3. Edit `provisioning_parameters.json` with Slurm-specific settings

4. Validate the configuration:
   ```bash
   hyp validate
   ```

5. Create the cluster:
   ```bash
   hyp create slurm-cluster --wait
   ```

6. Monitor cluster status:
   ```bash
   hyp describe slurm-cluster my-slurm-cluster
   ```

7. List cluster nodes:
   ```bash
   hyp list-nodes slurm-cluster my-slurm-cluster
   ```

## Error Handling

The CLI provides clear error messages for common issues:

- **Invalid credentials**: Displays remediation steps for AWS credential configuration
- **Insufficient permissions**: Lists the required IAM permissions
- **Rate limiting**: Automatically retries with exponential backoff (up to 3 times)
- **Cluster in transitional state**: Displays current state and advises waiting

## Differences from EKS Cluster Management

| Feature | EKS (cluster-stack) | Slurm (slurm-cluster) |
|---------|---------------------|----------------------|
| Backend | CloudFormation | SageMaker API |
| Init command | `hyp init cluster-stack` | `hyp init slurm-cluster` |
| Create command | `hyp create cluster-stack` | `hyp create slurm-cluster` |
| Node management | Via Kubernetes | Via SageMaker API |
| Orchestrator | Amazon EKS | Slurm workload manager |
