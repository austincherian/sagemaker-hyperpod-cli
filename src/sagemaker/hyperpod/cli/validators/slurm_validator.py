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
Validator for SageMaker HyperPod Slurm cluster configurations.

This module provides validation logic for Slurm cluster configurations,
including schema validation, ARN format validation, S3 URI validation,
instance group validation, VPC configuration validation, and cluster
state validation for update operations.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import ValidationError as PydanticValidationError

from sagemaker.hyperpod.cli.slurm_cluster_config import (
    IAM_ROLE_ARN_PATTERN,
    S3_URI_PATTERN,
    CLUSTER_NAME_PATTERN,
    SUBNET_ID_PATTERN,
    SECURITY_GROUP_ID_PATTERN,
    SlurmClusterConfig,
    InstanceGroupConfig,
    VpcConfig,
)
from sagemaker.hyperpod.cli.slurm_cluster_utils import (
    TRANSITIONAL_STATES,
    is_cluster_in_transitional_state,
)
from sagemaker.hyperpod.cli.validators.validator import Validator
from sagemaker.hyperpod.cli.utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class ValidationError:
    """
    Represents a validation error with field name and error message.

    Attributes:
        field: The name of the field that failed validation.
        message: A human-readable error message describing the validation failure.
        error_type: The type/category of the validation error.
    """
    field: str
    message: str
    error_type: str = "validation_error"

    def __str__(self) -> str:
        return f"{self.field}: {self.message}"


@dataclass
class ValidationResult:
    """
    Result of a validation operation.

    Attributes:
        is_valid: True if validation passed, False otherwise.
        errors: List of validation errors if validation failed.
    """
    is_valid: bool = True
    errors: List[ValidationError] = field(default_factory=list)

    def add_error(self, field: str, message: str, error_type: str = "validation_error") -> None:
        """Add a validation error to the result."""
        self.errors.append(ValidationError(field=field, message=message, error_type=error_type))
        self.is_valid = False

    def merge(self, other: "ValidationResult") -> None:
        """Merge another validation result into this one."""
        if not other.is_valid:
            self.is_valid = False
            self.errors.extend(other.errors)


class SlurmClusterValidator(Validator):
    """
    Validator for Slurm cluster configurations.

    This class provides comprehensive validation for Slurm cluster
    configurations, including schema validation, format validation
    for ARNs and S3 URIs, and state validation for update operations.
    """

    def __init__(self):
        """Initialize the SlurmClusterValidator."""
        super().__init__()

    def validate(self) -> ValidationResult:
        """
        Abstract validate method implementation.

        For SlurmClusterValidator, use validate_config() with a config object.
        """
        return ValidationResult(is_valid=True)

    def validate_config(
        self,
        config: Optional[SlurmClusterConfig] = None,
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate the entire Slurm cluster configuration.

        This method validates the configuration against the schema and
        performs additional semantic validations.

        Args:
            config: A SlurmClusterConfig object to validate.
            config_dict: A dictionary representation of the config to validate.
                        If provided, will attempt to parse into SlurmClusterConfig.

        Returns:
            ValidationResult containing validation status and any errors.

        Note:
            Either config or config_dict must be provided, but not both.
        """
        result = ValidationResult()

        if config is None and config_dict is None:
            result.add_error(
                field="config",
                message="Either config or config_dict must be provided",
                error_type="missing_input",
            )
            return result

        # If config_dict is provided, try to parse it
        if config_dict is not None:
            try:
                config = SlurmClusterConfig(**config_dict)
            except PydanticValidationError as e:
                # Extract field-specific errors from Pydantic validation
                for error in e.errors():
                    field_path = ".".join(str(loc) for loc in error["loc"])
                    result.add_error(
                        field=field_path,
                        message=error["msg"],
                        error_type="schema_violation",
                    )
                return result
            except Exception as e:
                result.add_error(
                    field="config",
                    message=f"Failed to parse configuration: {str(e)}",
                    error_type="parse_error",
                )
                return result

        # At this point, config should be a valid SlurmClusterConfig
        # Perform additional semantic validations
        if config is not None:
            # Validate instance groups
            ig_result = self.validate_instance_groups(config.instance_groups)
            result.merge(ig_result)

            # Validate VPC config
            vpc_result = self.validate_vpc_config(config.vpc_config)
            result.merge(vpc_result)

            # Validate Slurm config references
            slurm_result = self._validate_slurm_config_references(config)
            result.merge(slurm_result)

        return result

    def validate_arn_format(self, arn: str) -> bool:
        """
        Validate that a string is a valid IAM role ARN format.

        Args:
            arn: The string to validate.

        Returns:
            True if the string matches the IAM role ARN pattern, False otherwise.

        The expected format is: arn:aws:iam::<12-digit-account-id>:role/<role-name>
        """
        if not arn:
            return False
        return bool(IAM_ROLE_ARN_PATTERN.match(arn))

    def validate_s3_uri_format(self, uri: str) -> bool:
        """
        Validate that a string is a valid S3 URI format.

        Args:
            uri: The string to validate.

        Returns:
            True if the string matches the S3 URI pattern, False otherwise.

        The expected format is: s3://<bucket-name>/<path>
        where bucket name starts and ends with alphanumeric characters.
        """
        if not uri:
            return False
        return bool(S3_URI_PATTERN.match(uri))

    def validate_instance_groups(
        self,
        instance_groups: List[InstanceGroupConfig],
    ) -> ValidationResult:
        """
        Validate instance group configurations.

        This method validates:
        - Each instance group has a unique name
        - Execution roles have valid ARN format
        - Lifecycle config S3 URIs are valid
        - Instance counts are non-negative

        Args:
            instance_groups: List of InstanceGroupConfig objects to validate.

        Returns:
            ValidationResult containing validation status and any errors.
        """
        result = ValidationResult()

        if not instance_groups:
            result.add_error(
                field="instance_groups",
                message="At least one instance group is required",
                error_type="missing_required_field",
            )
            return result

        # Check for duplicate instance group names
        seen_names = set()
        for i, ig in enumerate(instance_groups):
            if ig.instance_group_name in seen_names:
                result.add_error(
                    field=f"instance_groups[{i}].instance_group_name",
                    message=f"Duplicate instance group name: '{ig.instance_group_name}'",
                    error_type="duplicate_value",
                )
            seen_names.add(ig.instance_group_name)

            # Validate execution role ARN format
            if not self.validate_arn_format(ig.execution_role):
                result.add_error(
                    field=f"instance_groups[{i}].execution_role",
                    message=f"Invalid IAM role ARN format: '{ig.execution_role}'. "
                            f"Expected format: arn:aws:iam::<account-id>:role/<role-name>",
                    error_type="invalid_format",
                )

            # Validate lifecycle config S3 URI
            if ig.lifecycle_config and ig.lifecycle_config.source_s3_uri:
                if not self.validate_s3_uri_format(ig.lifecycle_config.source_s3_uri):
                    result.add_error(
                        field=f"instance_groups[{i}].lifecycle_config.source_s3_uri",
                        message=f"Invalid S3 URI format: '{ig.lifecycle_config.source_s3_uri}'. "
                                f"Expected format: s3://bucket-name/path",
                        error_type="invalid_format",
                    )

            # Validate instance count
            if ig.instance_count < 0:
                result.add_error(
                    field=f"instance_groups[{i}].instance_count",
                    message=f"Instance count must be non-negative, got: {ig.instance_count}",
                    error_type="invalid_value",
                )

        return result

    def validate_vpc_config(self, vpc_config: VpcConfig) -> ValidationResult:
        """
        Validate VPC configuration.

        This method validates:
        - At least one subnet is specified
        - At least one security group is specified
        - Subnet IDs have valid format (subnet-<hex>)
        - Security group IDs have valid format (sg-<hex>)

        Args:
            vpc_config: VpcConfig object to validate.

        Returns:
            ValidationResult containing validation status and any errors.
        """
        result = ValidationResult()

        if not vpc_config:
            result.add_error(
                field="vpc_config",
                message="VPC configuration is required",
                error_type="missing_required_field",
            )
            return result

        # Validate subnets
        if not vpc_config.subnets:
            result.add_error(
                field="vpc_config.subnets",
                message="At least one subnet is required",
                error_type="missing_required_field",
            )
        else:
            for i, subnet_id in enumerate(vpc_config.subnets):
                if not SUBNET_ID_PATTERN.match(subnet_id):
                    result.add_error(
                        field=f"vpc_config.subnets[{i}]",
                        message=f"Invalid subnet ID format: '{subnet_id}'. "
                                f"Expected format: subnet-<hex-string>",
                        error_type="invalid_format",
                    )

        # Validate security groups
        if not vpc_config.security_group_ids:
            result.add_error(
                field="vpc_config.security_group_ids",
                message="At least one security group is required",
                error_type="missing_required_field",
            )
        else:
            for i, sg_id in enumerate(vpc_config.security_group_ids):
                if not SECURITY_GROUP_ID_PATTERN.match(sg_id):
                    result.add_error(
                        field=f"vpc_config.security_group_ids[{i}]",
                        message=f"Invalid security group ID format: '{sg_id}'. "
                                f"Expected format: sg-<hex-string>",
                        error_type="invalid_format",
                    )

        return result

    def validate_cluster_state_for_update(self, status: str) -> bool:
        """
        Check if a cluster can be updated in its current state.

        Clusters in transitional states (Creating, Updating, Deleting,
        RollingBack, SystemUpdating) cannot be updated.

        Args:
            status: The current cluster status.

        Returns:
            True if the cluster can be updated, False otherwise.
        """
        return not is_cluster_in_transitional_state(status)

    def validate_cluster_name(self, name: str) -> ValidationResult:
        """
        Validate cluster name format.

        Cluster names must:
        - Start with an alphanumeric character
        - Contain only alphanumeric characters and hyphens
        - Be between 1 and 63 characters

        Args:
            name: The cluster name to validate.

        Returns:
            ValidationResult containing validation status and any errors.
        """
        result = ValidationResult()

        if not name:
            result.add_error(
                field="cluster_name",
                message="Cluster name is required",
                error_type="missing_required_field",
            )
            return result

        if len(name) > 63:
            result.add_error(
                field="cluster_name",
                message=f"Cluster name must be at most 63 characters, got: {len(name)}",
                error_type="invalid_length",
            )

        if not CLUSTER_NAME_PATTERN.match(name):
            result.add_error(
                field="cluster_name",
                message=f"Invalid cluster name: '{name}'. "
                        f"Cluster name must start with an alphanumeric character "
                        f"and can only contain alphanumeric characters and hyphens.",
                error_type="invalid_format",
            )

        return result

    def _validate_slurm_config_references(
        self,
        config: SlurmClusterConfig,
    ) -> ValidationResult:
        """
        Validate that Slurm config references valid instance groups.

        This is an internal validation that checks:
        - controller_group references an existing instance group
        - login_group (if specified) references an existing instance group
        - All worker_groups reference existing instance groups

        Args:
            config: The SlurmClusterConfig to validate.

        Returns:
            ValidationResult containing validation status and any errors.
        """
        result = ValidationResult()

        instance_group_names = {ig.instance_group_name for ig in config.instance_groups}

        # Validate controller_group reference
        if config.slurm_config.controller_group not in instance_group_names:
            result.add_error(
                field="slurm_config.controller_group",
                message=f"Controller group '{config.slurm_config.controller_group}' "
                        f"does not match any instance group. "
                        f"Available groups: {sorted(instance_group_names)}",
                error_type="invalid_reference",
            )

        # Validate login_group reference if specified
        if (
            config.slurm_config.login_group
            and config.slurm_config.login_group not in instance_group_names
        ):
            result.add_error(
                field="slurm_config.login_group",
                message=f"Login group '{config.slurm_config.login_group}' "
                        f"does not match any instance group. "
                        f"Available groups: {sorted(instance_group_names)}",
                error_type="invalid_reference",
            )

        # Validate worker_groups references
        for i, worker_group in enumerate(config.slurm_config.worker_groups):
            if worker_group.instance_group_name not in instance_group_names:
                result.add_error(
                    field=f"slurm_config.worker_groups[{i}].instance_group_name",
                    message=f"Worker group instance_group_name "
                            f"'{worker_group.instance_group_name}' "
                            f"does not match any instance group. "
                            f"Available groups: {sorted(instance_group_names)}",
                    error_type="invalid_reference",
                )

        return result

    def get_validation_errors_summary(
        self,
        result: ValidationResult,
    ) -> str:
        """
        Get a human-readable summary of validation errors.

        Args:
            result: The ValidationResult to summarize.

        Returns:
            A formatted string containing all validation errors.
        """
        if result.is_valid:
            return "Configuration is valid."

        lines = ["Configuration validation failed with the following errors:"]
        for error in result.errors:
            lines.append(f"  - {error.field}: {error.message}")

        return "\n".join(lines)
