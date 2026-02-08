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
Pydantic models for Slurm cluster configuration.

This module defines the data models used for configuring SageMaker HyperPod
Slurm-orchestrated clusters. These models provide validation for cluster
configuration files and API requests.
"""

import re
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# Regex patterns for validation
CLUSTER_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9](-*[a-zA-Z0-9])*$")
IAM_ROLE_ARN_PATTERN = re.compile(r"^arn:aws:iam::\d{12}:role/.+$")
S3_URI_PATTERN = re.compile(r"^s3://[a-z0-9][a-z0-9.-]*[a-z0-9]/.+$")
SUBNET_ID_PATTERN = re.compile(r"^subnet-[a-f0-9]+$")
SECURITY_GROUP_ID_PATTERN = re.compile(r"^sg-[a-f0-9]+$")


class NodeRecovery(str, Enum):
    """Node recovery configuration options for HyperPod clusters."""

    AUTOMATIC = "Automatic"
    NONE = "None"


class EbsVolumeConfig(BaseModel):
    """Configuration for EBS volume attached to cluster instances."""

    model_config = ConfigDict(extra="forbid")

    volume_size_in_gb: int = Field(
        ...,
        ge=1,
        le=16384,
        description="Size of the EBS volume in GB (1-16384)",
    )


class InstanceStorageConfig(BaseModel):
    """Configuration for instance storage (EBS volumes)."""

    model_config = ConfigDict(extra="forbid")

    ebs_volume_config: Optional[EbsVolumeConfig] = Field(
        default=None,
        description="EBS volume configuration",
    )


class LifecycleConfig(BaseModel):
    """Lifecycle configuration for cluster instance provisioning."""

    model_config = ConfigDict(extra="forbid")

    source_s3_uri: str = Field(
        ...,
        min_length=1,
        description="S3 URI containing lifecycle scripts",
    )
    on_create: str = Field(
        ...,
        min_length=1,
        description="Script to run during instance creation",
    )

    @field_validator("source_s3_uri")
    @classmethod
    def validate_s3_uri(cls, v: str) -> str:
        """Validate that source_s3_uri is a valid S3 URI format."""
        if not S3_URI_PATTERN.match(v):
            raise ValueError(
                f"Invalid S3 URI format: '{v}'. "
                "Expected format: s3://bucket-name/path"
            )
        return v


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
        ...,
        min_length=1,
        description="EC2 instance type (e.g., ml.p4d.24xlarge)",
    )
    instance_count: int = Field(
        ...,
        ge=0,
        description="Number of instances in the group",
    )
    execution_role: str = Field(
        ...,
        min_length=1,
        description="IAM role ARN for instance execution",
    )
    lifecycle_config: LifecycleConfig = Field(
        ...,
        description="Lifecycle configuration for the instance group",
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
    on_start_deep_health_checks: Optional[List[str]] = Field(
        default=None,
        description="Deep health checks to run on instance start",
    )

    @field_validator("execution_role")
    @classmethod
    def validate_execution_role_arn(cls, v: str) -> str:
        """Validate that execution_role is a valid IAM role ARN format."""
        if not IAM_ROLE_ARN_PATTERN.match(v):
            raise ValueError(
                f"Invalid IAM role ARN format: '{v}'. "
                "Expected format: arn:aws:iam::<account-id>:role/<role-name>"
            )
        return v


class VpcConfig(BaseModel):
    """VPC configuration for the cluster."""

    model_config = ConfigDict(extra="forbid")

    subnets: List[str] = Field(
        ...,
        min_length=1,
        description="List of subnet IDs for the cluster",
    )
    security_group_ids: List[str] = Field(
        ...,
        min_length=1,
        description="List of security group IDs for the cluster",
    )

    @field_validator("subnets")
    @classmethod
    def validate_subnet_ids(cls, v: List[str]) -> List[str]:
        """Validate that all subnet IDs have valid format."""
        for subnet_id in v:
            if not SUBNET_ID_PATTERN.match(subnet_id):
                raise ValueError(
                    f"Invalid subnet ID format: '{subnet_id}'. "
                    "Expected format: subnet-<hex-string>"
                )
        return v

    @field_validator("security_group_ids")
    @classmethod
    def validate_security_group_ids(cls, v: List[str]) -> List[str]:
        """Validate that all security group IDs have valid format."""
        for sg_id in v:
            if not SECURITY_GROUP_ID_PATTERN.match(sg_id):
                raise ValueError(
                    f"Invalid security group ID format: '{sg_id}'. "
                    "Expected format: sg-<hex-string>"
                )
        return v


class Tag(BaseModel):
    """AWS resource tag."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Tag key",
    )
    value: str = Field(
        ...,
        max_length=256,
        description="Tag value",
    )


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
        description="Slurm partition name for this worker group",
    )


class SlurmConfig(BaseModel):
    """Slurm-specific configuration for the cluster."""

    model_config = ConfigDict(extra="forbid")

    controller_group: str = Field(
        ...,
        min_length=1,
        description="Instance group name for the Slurm controller",
    )
    login_group: Optional[str] = Field(
        default=None,
        description="Instance group name for Slurm login nodes",
    )
    worker_groups: List[WorkerGroupConfig] = Field(
        ...,
        min_length=1,
        description="List of worker group configurations",
    )
    fsx_dns_name: Optional[str] = Field(
        default=None,
        description="FSx file system DNS name for shared storage",
    )
    fsx_mountname: Optional[str] = Field(
        default=None,
        description="FSx mount name",
    )


class SlurmClusterConfig(BaseModel):
    """
    Complete configuration for a SageMaker HyperPod Slurm cluster.

    This model represents the full configuration needed to create or
    manage a Slurm-orchestrated HyperPod cluster.
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
        ...,
        min_length=1,
        max_length=63,
        description="Name of the HyperPod cluster",
    )
    region: Optional[str] = Field(
        default=None,
        description="AWS region for the cluster",
    )
    instance_groups: List[InstanceGroupConfig] = Field(
        ...,
        min_length=1,
        description="List of instance group configurations",
    )
    vpc_config: VpcConfig = Field(
        ...,
        description="VPC configuration for the cluster",
    )
    node_recovery: NodeRecovery = Field(
        default=NodeRecovery.AUTOMATIC,
        description="Node recovery setting (Automatic or None)",
    )
    tags: Optional[List[Tag]] = Field(
        default=None,
        description="AWS resource tags for the cluster",
    )
    slurm_config: SlurmConfig = Field(
        ...,
        description="Slurm-specific configuration",
    )

    @field_validator("cluster_name")
    @classmethod
    def validate_cluster_name(cls, v: str) -> str:
        """
        Validate that cluster_name matches the required pattern.

        Cluster names must start with an alphanumeric character and can
        contain alphanumeric characters and hyphens.
        """
        if not CLUSTER_NAME_PATTERN.match(v):
            raise ValueError(
                f"Invalid cluster name: '{v}'. "
                "Cluster name must start with an alphanumeric character "
                "and can only contain alphanumeric characters and hyphens."
            )
        return v

    @model_validator(mode="after")
    def validate_instance_group_references(self) -> "SlurmClusterConfig":
        """
        Validate that Slurm config references valid instance groups.

        Ensures that controller_group, login_group, and worker_groups
        all reference instance groups that exist in the configuration.
        """
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
                f"does not match any instance group. "
                f"Available groups: {sorted(instance_group_names)}"
            )

        # Validate worker_groups references
        for worker_group in self.slurm_config.worker_groups:
            if worker_group.instance_group_name not in instance_group_names:
                raise ValueError(
                    f"Worker group instance_group_name "
                    f"'{worker_group.instance_group_name}' "
                    f"does not match any instance group. "
                    f"Available groups: {sorted(instance_group_names)}"
                )

        return self

    def to_create_cluster_request(self) -> Dict[str, Any]:
        """
        Convert the configuration to a SageMaker CreateCluster API request.

        Returns:
            Dict containing the parameters for the CreateCluster API call.
        """
        request: Dict[str, Any] = {
            "ClusterName": self.cluster_name,
            "InstanceGroups": [
                {
                    "InstanceGroupName": ig.instance_group_name,
                    "InstanceType": ig.instance_type,
                    "InstanceCount": ig.instance_count,
                    "ExecutionRole": ig.execution_role,
                    "LifeCycleConfig": {
                        "SourceS3Uri": ig.lifecycle_config.source_s3_uri,
                        "OnCreate": ig.lifecycle_config.on_create,
                    },
                    **(
                        {
                            "InstanceStorageConfigs": [
                                {
                                    "EbsVolumeConfig": {
                                        "VolumeSizeInGB": isc.ebs_volume_config.volume_size_in_gb
                                    }
                                }
                                for isc in ig.instance_storage_configs
                                if isc.ebs_volume_config
                            ]
                        }
                        if ig.instance_storage_configs
                        else {}
                    ),
                    **(
                        {"ThreadsPerCore": ig.threads_per_core}
                        if ig.threads_per_core
                        else {}
                    ),
                    **(
                        {"OnStartDeepHealthChecks": ig.on_start_deep_health_checks}
                        if ig.on_start_deep_health_checks
                        else {}
                    ),
                }
                for ig in self.instance_groups
            ],
            "VpcConfig": {
                "Subnets": self.vpc_config.subnets,
                "SecurityGroupIds": self.vpc_config.security_group_ids,
            },
        }

        if self.node_recovery:
            request["NodeRecovery"] = self.node_recovery.value

        if self.tags:
            request["Tags"] = [
                {"Key": tag.key, "Value": tag.value} for tag in self.tags
            ]

        return request


# Utility functions for ARN and S3 URI validation (standalone validators)


def validate_arn_format(arn: str) -> bool:
    """
    Validate that a string is a valid IAM role ARN format.

    Args:
        arn: The string to validate.

    Returns:
        True if the string matches the IAM role ARN pattern, False otherwise.
    """
    return bool(IAM_ROLE_ARN_PATTERN.match(arn))


def validate_s3_uri_format(uri: str) -> bool:
    """
    Validate that a string is a valid S3 URI format.

    Args:
        uri: The string to validate.

    Returns:
        True if the string matches the S3 URI pattern, False otherwise.
    """
    return bool(S3_URI_PATTERN.match(uri))


def validate_cluster_name_format(name: str) -> bool:
    """
    Validate that a string is a valid cluster name format.

    Args:
        name: The string to validate.

    Returns:
        True if the string matches the cluster name pattern, False otherwise.
    """
    return bool(CLUSTER_NAME_PATTERN.match(name))
