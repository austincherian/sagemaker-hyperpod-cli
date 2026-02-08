# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""
Pydantic models for Slurm cluster template configuration (v1.0).

This module defines the template configuration model used for scaffolding
Slurm cluster configuration files via `hyp init slurm-cluster`.
"""

import re
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# Regex patterns for validation
CLUSTER_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9](-*[a-zA-Z0-9])*$")
IAM_ROLE_ARN_PATTERN = re.compile(r"^arn:aws:iam::\d{12}:role/.+$")
S3_URI_PATTERN = re.compile(r"^s3://[a-z0-9][a-z0-9.-]*[a-z0-9]/.+$")


class NodeRecovery(str, Enum):
    """Node recovery configuration options."""

    AUTOMATIC = "Automatic"
    NONE = "None"


class EbsVolumeConfig(BaseModel):
    """EBS volume configuration for instance storage."""

    model_config = ConfigDict(extra="forbid")

    volume_size_in_gb: int = Field(
        default=500,
        ge=1,
        le=16384,
        description="Size of the EBS volume in GB",
    )


class InstanceStorageConfig(BaseModel):
    """Instance storage configuration."""

    model_config = ConfigDict(extra="forbid")

    ebs_volume_config: Optional[EbsVolumeConfig] = Field(
        default=None,
        description="EBS volume configuration",
    )


class LifecycleConfig(BaseModel):
    """Lifecycle configuration for instance provisioning."""

    model_config = ConfigDict(extra="forbid")

    source_s3_uri: str = Field(
        default="s3://your-bucket/lifecycle-scripts",
        description="S3 URI containing lifecycle scripts",
    )
    on_create: str = Field(
        default="on_create.sh",
        description="Script to run during instance creation",
    )


class InstanceGroupConfig(BaseModel):
    """Configuration for a cluster instance group."""

    model_config = ConfigDict(extra="forbid")

    instance_group_name: str = Field(
        ...,
        min_length=1,
        max_length=63,
        description="Name of the instance group",
    )
    instance_type: str = Field(
        default="ml.t3.medium",
        description="EC2 instance type",
    )
    instance_count: int = Field(
        default=1,
        ge=0,
        description="Number of instances in the group",
    )
    execution_role: str = Field(
        default="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
        description="IAM role ARN for instance execution",
    )
    lifecycle_config: LifecycleConfig = Field(
        default_factory=LifecycleConfig,
        description="Lifecycle configuration",
    )
    instance_storage_configs: Optional[List[InstanceStorageConfig]] = Field(
        default=None,
        description="Storage configurations for instances",
    )
    threads_per_core: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of threads per CPU core",
    )


class VpcConfig(BaseModel):
    """VPC configuration for the cluster."""

    model_config = ConfigDict(extra="forbid")

    subnets: List[str] = Field(
        default_factory=lambda: ["subnet-xxxxxxxxxxxxxxxxx"],
        description="List of subnet IDs",
    )
    security_group_ids: List[str] = Field(
        default_factory=lambda: ["sg-xxxxxxxxxxxxxxxxx"],
        description="List of security group IDs",
    )


class Tag(BaseModel):
    """AWS resource tag."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(..., min_length=1, max_length=128, description="Tag key")
    value: str = Field(..., max_length=256, description="Tag value")


class WorkerGroupConfig(BaseModel):
    """Configuration for a Slurm worker group."""

    model_config = ConfigDict(extra="forbid")

    instance_group_name: str = Field(
        ...,
        min_length=1,
        description="Name of the instance group for this worker group",
    )
    partition_name: str = Field(
        ...,
        min_length=1,
        description="Slurm partition name",
    )


class SlurmConfig(BaseModel):
    """Slurm-specific configuration."""

    model_config = ConfigDict(extra="forbid")

    controller_group: str = Field(
        default="controller",
        description="Instance group name for the Slurm controller",
    )
    login_group: Optional[str] = Field(
        default=None,
        description="Instance group name for Slurm login nodes",
    )
    worker_groups: List[WorkerGroupConfig] = Field(
        default_factory=lambda: [
            WorkerGroupConfig(instance_group_name="worker", partition_name="dev")
        ],
        description="List of worker group configurations",
    )
    fsx_dns_name: Optional[str] = Field(
        default=None,
        description="FSx file system DNS name",
    )
    fsx_mountname: Optional[str] = Field(
        default=None,
        description="FSx mount name",
    )


class SlurmClusterTemplateConfig(BaseModel):
    """
    Template configuration for scaffolding Slurm cluster config files.

    This model is used by `hyp init slurm-cluster` to generate the initial
    config.yaml file with sensible defaults.
    """

    model_config = ConfigDict(extra="forbid")

    version: str = Field(
        default="1.0",
        description="Configuration schema version",
    )
    template: str = Field(
        default="slurm-cluster",
        description="Template type identifier",
    )
    cluster_name: str = Field(
        default="my-slurm-cluster",
        min_length=1,
        max_length=63,
        description="Name of the HyperPod cluster",
    )
    region: Optional[str] = Field(
        default=None,
        description="AWS region for the cluster",
    )
    instance_groups: List[InstanceGroupConfig] = Field(
        default_factory=lambda: [
            InstanceGroupConfig(
                instance_group_name="controller",
                instance_type="ml.t3.medium",
                instance_count=1,
            ),
            InstanceGroupConfig(
                instance_group_name="worker",
                instance_type="ml.p4d.24xlarge",
                instance_count=2,
            ),
        ],
        description="List of instance group configurations",
    )
    vpc_config: VpcConfig = Field(
        default_factory=VpcConfig,
        description="VPC configuration",
    )
    node_recovery: NodeRecovery = Field(
        default=NodeRecovery.AUTOMATIC,
        description="Node recovery setting",
    )
    tags: Optional[List[Tag]] = Field(
        default=None,
        description="AWS resource tags",
    )
    slurm_config: SlurmConfig = Field(
        default_factory=SlurmConfig,
        description="Slurm-specific configuration",
    )

    @field_validator("cluster_name")
    @classmethod
    def validate_cluster_name(cls, v: str) -> str:
        """Validate cluster name format."""
        if not CLUSTER_NAME_PATTERN.match(v):
            raise ValueError(
                f"Invalid cluster name: '{v}'. "
                "Must start with alphanumeric and contain only alphanumeric and hyphens."
            )
        return v

    @model_validator(mode="after")
    def validate_instance_group_references(self) -> "SlurmClusterTemplateConfig":
        """Validate that Slurm config references valid instance groups."""
        instance_group_names = {ig.instance_group_name for ig in self.instance_groups}

        # Validate controller_group reference
        if self.slurm_config.controller_group not in instance_group_names:
            raise ValueError(
                f"Slurm controller_group '{self.slurm_config.controller_group}' "
                f"does not match any instance group. "
                f"Available groups: {sorted(instance_group_names)}"
            )

        # Validate login_group reference if specified
        if (
            self.slurm_config.login_group
            and self.slurm_config.login_group not in instance_group_names
        ):
            raise ValueError(
                f"Slurm login_group '{self.slurm_config.login_group}' "
                f"does not match any instance group."
            )

        # Validate worker_groups references
        for worker_group in self.slurm_config.worker_groups:
            if worker_group.instance_group_name not in instance_group_names:
                raise ValueError(
                    f"Worker group '{worker_group.instance_group_name}' "
                    f"does not match any instance group."
                )

        return self

    def to_config_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return self.model_dump(exclude_none=True, by_alias=True)
