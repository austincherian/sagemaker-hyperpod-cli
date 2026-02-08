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
Property-based tests for Slurm cluster configuration.

These tests use Hypothesis to verify correctness properties for
configuration serialization, validation, and format checking.

Properties tested:
- Property 1: Config round-trip consistency (Requirements 12.1, 12.2, 12.3)
- Property 2: Invalid config rejection (Requirements 1.2, 3.4, 9.1)
- Property 3: ARN format validation (Requirements 9.2)
- Property 4: S3 URI format validation (Requirements 9.3)
- Property 12: Validation error specificity (Requirements 9.5)
"""

import string
import yaml
from typing import Any, Dict, List, Optional

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st
from pydantic import ValidationError as PydanticValidationError

from sagemaker.hyperpod.cli.slurm_cluster_config import (
    EbsVolumeConfig,
    InstanceGroupConfig,
    InstanceStorageConfig,
    LifecycleConfig,
    NodeRecovery,
    SlurmClusterConfig,
    SlurmConfig,
    Tag,
    VpcConfig,
    WorkerGroupConfig,
    validate_arn_format,
    validate_s3_uri_format,
    validate_cluster_name_format,
    IAM_ROLE_ARN_PATTERN,
    S3_URI_PATTERN,
    CLUSTER_NAME_PATTERN,
)
from sagemaker.hyperpod.cli.validators.slurm_validator import (
    SlurmClusterValidator,
    ValidationResult,
)


# =============================================================================
# Hypothesis Strategies for generating test data
# =============================================================================


@st.composite
def valid_cluster_names(draw) -> str:
    """Generate valid cluster names matching the pattern ^[a-zA-Z0-9](-*[a-zA-Z0-9])*$"""
    # Start with alphanumeric
    first_char = draw(st.sampled_from(string.ascii_letters + string.digits))
    
    # Middle can be alphanumeric or hyphens, but not consecutive hyphens
    middle_length = draw(st.integers(min_value=0, max_value=30))
    middle = ""
    last_was_hyphen = False
    
    for _ in range(middle_length):
        if last_was_hyphen:
            # Must be alphanumeric after hyphen
            char = draw(st.sampled_from(string.ascii_letters + string.digits))
            last_was_hyphen = False
        else:
            # Can be alphanumeric or hyphen
            char = draw(st.sampled_from(string.ascii_letters + string.digits + "-"))
            last_was_hyphen = (char == "-")
        middle += char
    
    # End with alphanumeric if we have middle chars and last was hyphen
    if middle and middle[-1] == "-":
        middle = middle[:-1] + draw(st.sampled_from(string.ascii_letters + string.digits))
    
    result = first_char + middle
    # Ensure max length
    return result[:63]


@st.composite
def invalid_cluster_names(draw) -> str:
    """Generate invalid cluster names that don't match the pattern."""
    strategy = draw(st.sampled_from([
        # Starts with hyphen
        st.just("-invalid-name"),
        # Contains special characters
        st.just("invalid_name"),
        st.just("invalid.name"),
        st.just("invalid@name"),
        st.just("invalid name"),
        # Empty string
        st.just(""),
        # Too long (over 63 chars)
        st.just("a" * 64),
        # Ends with hyphen
        st.just("invalid-"),
        # Consecutive hyphens
        st.just("invalid--name"),
    ]))
    return draw(strategy)


@st.composite
def valid_arn_strings(draw) -> str:
    """Generate valid IAM role ARN strings."""
    account_id = draw(st.text(alphabet=string.digits, min_size=12, max_size=12))
    role_name = draw(st.text(
        alphabet=string.ascii_letters + string.digits + "-_",
        min_size=1,
        max_size=30
    ))
    return f"arn:aws:iam::{account_id}:role/{role_name}"


@st.composite
def invalid_arn_strings(draw) -> str:
    """Generate invalid ARN strings."""
    strategy = draw(st.sampled_from([
        # Missing arn: prefix
        st.just("aws:iam::123456789012:role/MyRole"),
        # Wrong service
        st.just("arn:aws:s3::123456789012:role/MyRole"),
        # Wrong account ID length
        st.just("arn:aws:iam::12345:role/MyRole"),
        # Missing role path
        st.just("arn:aws:iam::123456789012:role/"),
        # Non-numeric account ID
        st.just("arn:aws:iam::abcdefghijkl:role/MyRole"),
        # Empty string
        st.just(""),
        # Random string
        st.text(min_size=1, max_size=20),
    ]))
    return draw(strategy)


@st.composite
def valid_s3_uri_strings(draw) -> str:
    """Generate valid S3 URI strings."""
    # Bucket name: starts and ends with alphanumeric, can contain dots and hyphens
    bucket_start = draw(st.sampled_from(string.ascii_lowercase + string.digits))
    bucket_middle = draw(st.text(
        alphabet=string.ascii_lowercase + string.digits + ".-",
        min_size=1,
        max_size=30
    ))
    bucket_end = draw(st.sampled_from(string.ascii_lowercase + string.digits))
    bucket_name = bucket_start + bucket_middle + bucket_end
    
    # Path: at least one character
    path = draw(st.text(
        alphabet=string.ascii_letters + string.digits + "/-_.",
        min_size=1,
        max_size=50
    ))
    
    return f"s3://{bucket_name}/{path}"


@st.composite
def invalid_s3_uri_strings(draw) -> str:
    """Generate invalid S3 URI strings."""
    strategy = draw(st.sampled_from([
        # Missing s3:// prefix
        st.just("bucket/path"),
        # Wrong prefix
        st.just("http://bucket/path"),
        # Bucket starts with hyphen
        st.just("s3://-bucket/path"),
        # Bucket ends with hyphen
        st.just("s3://bucket-/path"),
        # Missing path
        st.just("s3://bucket"),
        st.just("s3://bucket/"),
        # Empty string
        st.just(""),
        # Uppercase in bucket (technically invalid for S3)
        st.just("s3://MyBucket/path"),
    ]))
    return draw(strategy)


@st.composite
def valid_subnet_ids(draw) -> str:
    """Generate valid subnet IDs."""
    hex_chars = draw(st.text(alphabet="0123456789abcdef", min_size=8, max_size=17))
    return f"subnet-{hex_chars}"


@st.composite
def valid_security_group_ids(draw) -> str:
    """Generate valid security group IDs."""
    hex_chars = draw(st.text(alphabet="0123456789abcdef", min_size=8, max_size=17))
    return f"sg-{hex_chars}"


@st.composite
def lifecycle_config_strategy(draw) -> LifecycleConfig:
    """Generate valid LifecycleConfig objects."""
    s3_uri = draw(valid_s3_uri_strings())
    on_create = draw(st.text(
        alphabet=string.ascii_letters + string.digits + "_-.",
        min_size=1,
        max_size=50
    ))
    return LifecycleConfig(source_s3_uri=s3_uri, on_create=on_create)


@st.composite
def instance_group_config_strategy(draw, name: Optional[str] = None) -> InstanceGroupConfig:
    """Generate valid InstanceGroupConfig objects."""
    if name is None:
        name = draw(st.text(
            alphabet=string.ascii_letters + string.digits + "-",
            min_size=1,
            max_size=30
        ))
        # Ensure name starts with alphanumeric
        if name and name[0] == "-":
            name = "a" + name
    
    instance_type = draw(st.sampled_from([
        "ml.t3.medium",
        "ml.m5.xlarge",
        "ml.m5.2xlarge",
        "ml.p4d.24xlarge",
        "ml.p5.48xlarge",
    ]))
    
    instance_count = draw(st.integers(min_value=0, max_value=100))
    execution_role = draw(valid_arn_strings())
    lifecycle_config = draw(lifecycle_config_strategy())
    
    return InstanceGroupConfig(
        instance_group_name=name,
        instance_type=instance_type,
        instance_count=instance_count,
        execution_role=execution_role,
        lifecycle_config=lifecycle_config,
    )


@st.composite
def vpc_config_strategy(draw) -> VpcConfig:
    """Generate valid VpcConfig objects."""
    num_subnets = draw(st.integers(min_value=1, max_value=3))
    subnets = [draw(valid_subnet_ids()) for _ in range(num_subnets)]
    
    num_sgs = draw(st.integers(min_value=1, max_value=3))
    security_groups = [draw(valid_security_group_ids()) for _ in range(num_sgs)]
    
    return VpcConfig(subnets=subnets, security_group_ids=security_groups)


@st.composite
def tag_strategy(draw) -> Tag:
    """Generate valid Tag objects."""
    key = draw(st.text(
        alphabet=string.ascii_letters + string.digits + "-_",
        min_size=1,
        max_size=50
    ))
    value = draw(st.text(
        alphabet=string.ascii_letters + string.digits + "-_ ",
        min_size=0,
        max_size=100
    ))
    return Tag(key=key, value=value)


@st.composite
def worker_group_config_strategy(draw, instance_group_name: str) -> WorkerGroupConfig:
    """Generate valid WorkerGroupConfig objects."""
    partition_name = draw(st.text(
        alphabet=string.ascii_letters + string.digits + "-_",
        min_size=1,
        max_size=30
    ))
    return WorkerGroupConfig(
        instance_group_name=instance_group_name,
        partition_name=partition_name,
    )


@st.composite
def slurm_config_strategy(draw, instance_group_names: List[str]) -> SlurmConfig:
    """Generate valid SlurmConfig objects referencing given instance groups."""
    assume(len(instance_group_names) >= 1)
    
    controller_group = instance_group_names[0]
    
    # Create worker groups from remaining instance groups
    worker_groups = []
    for name in instance_group_names[1:] if len(instance_group_names) > 1 else [instance_group_names[0]]:
        worker_groups.append(draw(worker_group_config_strategy(name)))
    
    # Ensure at least one worker group
    if not worker_groups:
        worker_groups.append(draw(worker_group_config_strategy(controller_group)))
    
    return SlurmConfig(
        controller_group=controller_group,
        worker_groups=worker_groups,
    )


@st.composite
def slurm_cluster_config_strategy(draw) -> SlurmClusterConfig:
    """Generate valid SlurmClusterConfig objects."""
    cluster_name = draw(valid_cluster_names())
    assume(len(cluster_name) >= 1)
    
    # Generate instance groups
    num_groups = draw(st.integers(min_value=1, max_value=3))
    instance_group_names = []
    instance_groups = []
    
    for i in range(num_groups):
        name = f"group{i}"
        instance_group_names.append(name)
        instance_groups.append(draw(instance_group_config_strategy(name=name)))
    
    vpc_config = draw(vpc_config_strategy())
    node_recovery = draw(st.sampled_from([NodeRecovery.AUTOMATIC, NodeRecovery.NONE]))
    
    # Generate tags (optional)
    include_tags = draw(st.booleans())
    tags = None
    if include_tags:
        num_tags = draw(st.integers(min_value=1, max_value=3))
        tags = [draw(tag_strategy()) for _ in range(num_tags)]
    
    slurm_config = draw(slurm_config_strategy(instance_group_names))
    
    return SlurmClusterConfig(
        cluster_name=cluster_name,
        instance_groups=instance_groups,
        vpc_config=vpc_config,
        node_recovery=node_recovery,
        tags=tags,
        slurm_config=slurm_config,
    )


# =============================================================================
# Property 1: Config round-trip consistency
# Feature: hyperpod-slurm-cluster-crud, Property 1: Config round-trip consistency
# Validates: Requirements 12.1, 12.2, 12.3
# =============================================================================


class TestConfigRoundTrip:
    """
    Property 1: Config round-trip consistency
    
    *For any* valid SlurmClusterConfig object, serializing it to YAML and then
    deserializing the YAML back to a SlurmClusterConfig object SHALL produce
    a semantically equivalent configuration.
    
    **Validates: Requirements 12.1, 12.2, 12.3**
    """

    @given(config=slurm_cluster_config_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_config_yaml_round_trip(self, config: SlurmClusterConfig):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 1: Config round-trip consistency
        **Validates: Requirements 12.1, 12.2, 12.3**
        
        Serializing a config to YAML and deserializing should produce equivalent config.
        """
        # Serialize to dict then to YAML (use mode='json' to serialize enums as strings)
        config_dict = config.model_dump(mode='json')
        yaml_str = yaml.dump(config_dict, default_flow_style=False)
        
        # Deserialize from YAML
        loaded_dict = yaml.safe_load(yaml_str)
        loaded_config = SlurmClusterConfig(**loaded_dict)
        
        # Verify equivalence
        assert loaded_config.cluster_name == config.cluster_name
        assert loaded_config.version == config.version
        assert loaded_config.template == config.template
        assert loaded_config.node_recovery == config.node_recovery
        assert len(loaded_config.instance_groups) == len(config.instance_groups)
        assert loaded_config.vpc_config.subnets == config.vpc_config.subnets
        assert loaded_config.vpc_config.security_group_ids == config.vpc_config.security_group_ids
        assert loaded_config.slurm_config.controller_group == config.slurm_config.controller_group

    @given(config=slurm_cluster_config_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_config_dict_round_trip(self, config: SlurmClusterConfig):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 1: Config round-trip consistency
        **Validates: Requirements 12.1, 12.2, 12.3**
        
        Converting config to dict and back should produce equivalent config.
        """
        # Convert to dict
        config_dict = config.model_dump()
        
        # Convert back to config
        loaded_config = SlurmClusterConfig(**config_dict)
        
        # Verify full equivalence using model comparison
        assert loaded_config.model_dump() == config.model_dump()


# =============================================================================
# Property 2: Invalid config rejection
# Feature: hyperpod-slurm-cluster-crud, Property 2: Invalid config rejection
# Validates: Requirements 1.2, 3.4, 9.1
# =============================================================================


class TestInvalidConfigRejection:
    """
    Property 2: Invalid config rejection
    
    *For any* configuration with schema violations (missing required fields,
    invalid types, out-of-range values), the validator SHALL return at least
    one validation error specifying the invalid field.
    
    **Validates: Requirements 1.2, 3.4, 9.1**
    """

    @given(cluster_name=invalid_cluster_names())
    @settings(max_examples=100)
    def test_invalid_cluster_name_rejected(self, cluster_name: str):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 2: Invalid config rejection
        **Validates: Requirements 1.2, 3.4, 9.1**
        
        Invalid cluster names should be rejected with validation error.
        """
        # Skip empty strings as they're handled differently
        assume(cluster_name != "")
        
        config_dict = {
            "cluster_name": cluster_name,
            "instance_groups": [
                {
                    "instance_group_name": "controller",
                    "instance_type": "ml.t3.medium",
                    "instance_count": 1,
                    "execution_role": "arn:aws:iam::123456789012:role/TestRole",
                    "lifecycle_config": {
                        "source_s3_uri": "s3://my-bucket/scripts",
                        "on_create": "setup.sh",
                    },
                }
            ],
            "vpc_config": {
                "subnets": ["subnet-12345678"],
                "security_group_ids": ["sg-12345678"],
            },
            "slurm_config": {
                "controller_group": "controller",
                "worker_groups": [
                    {"instance_group_name": "controller", "partition_name": "default"}
                ],
            },
        }
        
        validator = SlurmClusterValidator()
        result = validator.validate_config(config_dict=config_dict)
        
        # Should either fail validation or raise exception during parsing
        # Check if cluster_name pattern is invalid
        if not CLUSTER_NAME_PATTERN.match(cluster_name) or len(cluster_name) > 63:
            # Either validation fails or Pydantic raises
            if result.is_valid:
                # Try to create the config - should raise
                try:
                    SlurmClusterConfig(**config_dict)
                    # If we get here with an invalid name, that's a bug
                    assert CLUSTER_NAME_PATTERN.match(cluster_name), \
                        f"Invalid cluster name '{cluster_name}' was accepted"
                except (PydanticValidationError, ValueError):
                    pass  # Expected

    @given(arn=invalid_arn_strings())
    @settings(max_examples=100)
    def test_invalid_execution_role_rejected(self, arn: str):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 2: Invalid config rejection
        **Validates: Requirements 1.2, 3.4, 9.1**
        
        Invalid execution role ARNs should be rejected.
        """
        config_dict = {
            "cluster_name": "test-cluster",
            "instance_groups": [
                {
                    "instance_group_name": "controller",
                    "instance_type": "ml.t3.medium",
                    "instance_count": 1,
                    "execution_role": arn,
                    "lifecycle_config": {
                        "source_s3_uri": "s3://my-bucket/scripts",
                        "on_create": "setup.sh",
                    },
                }
            ],
            "vpc_config": {
                "subnets": ["subnet-12345678"],
                "security_group_ids": ["sg-12345678"],
            },
            "slurm_config": {
                "controller_group": "controller",
                "worker_groups": [
                    {"instance_group_name": "controller", "partition_name": "default"}
                ],
            },
        }
        
        # Should fail validation or raise during parsing
        try:
            config = SlurmClusterConfig(**config_dict)
            # If parsing succeeded, validation should fail
            validator = SlurmClusterValidator()
            result = validator.validate_config(config=config)
            
            if not IAM_ROLE_ARN_PATTERN.match(arn):
                assert not result.is_valid, \
                    f"Invalid ARN '{arn}' was accepted"
                # Verify error mentions the field
                error_fields = [e.field for e in result.errors]
                assert any("execution_role" in f for f in error_fields), \
                    f"Error should mention execution_role field"
        except (PydanticValidationError, ValueError):
            pass  # Expected for invalid ARNs

    @given(s3_uri=invalid_s3_uri_strings())
    @settings(max_examples=100)
    def test_invalid_s3_uri_rejected(self, s3_uri: str):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 2: Invalid config rejection
        **Validates: Requirements 1.2, 3.4, 9.1**
        
        Invalid S3 URIs should be rejected.
        """
        config_dict = {
            "cluster_name": "test-cluster",
            "instance_groups": [
                {
                    "instance_group_name": "controller",
                    "instance_type": "ml.t3.medium",
                    "instance_count": 1,
                    "execution_role": "arn:aws:iam::123456789012:role/TestRole",
                    "lifecycle_config": {
                        "source_s3_uri": s3_uri,
                        "on_create": "setup.sh",
                    },
                }
            ],
            "vpc_config": {
                "subnets": ["subnet-12345678"],
                "security_group_ids": ["sg-12345678"],
            },
            "slurm_config": {
                "controller_group": "controller",
                "worker_groups": [
                    {"instance_group_name": "controller", "partition_name": "default"}
                ],
            },
        }
        
        # Should fail validation or raise during parsing
        try:
            config = SlurmClusterConfig(**config_dict)
            # If parsing succeeded, validation should fail
            validator = SlurmClusterValidator()
            result = validator.validate_config(config=config)
            
            if not S3_URI_PATTERN.match(s3_uri):
                assert not result.is_valid, \
                    f"Invalid S3 URI '{s3_uri}' was accepted"
        except (PydanticValidationError, ValueError):
            pass  # Expected for invalid S3 URIs

    def test_missing_required_field_rejected(self):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 2: Invalid config rejection
        **Validates: Requirements 1.2, 3.4, 9.1**
        
        Missing required fields should be rejected with specific error.
        """
        # Missing cluster_name
        config_dict = {
            "instance_groups": [
                {
                    "instance_group_name": "controller",
                    "instance_type": "ml.t3.medium",
                    "instance_count": 1,
                    "execution_role": "arn:aws:iam::123456789012:role/TestRole",
                    "lifecycle_config": {
                        "source_s3_uri": "s3://my-bucket/scripts",
                        "on_create": "setup.sh",
                    },
                }
            ],
            "vpc_config": {
                "subnets": ["subnet-12345678"],
                "security_group_ids": ["sg-12345678"],
            },
            "slurm_config": {
                "controller_group": "controller",
                "worker_groups": [
                    {"instance_group_name": "controller", "partition_name": "default"}
                ],
            },
        }
        
        validator = SlurmClusterValidator()
        result = validator.validate_config(config_dict=config_dict)
        
        assert not result.is_valid
        error_fields = [e.field for e in result.errors]
        assert any("cluster_name" in f for f in error_fields)

    def test_invalid_instance_count_rejected(self):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 2: Invalid config rejection
        **Validates: Requirements 1.2, 3.4, 9.1**
        
        Negative instance count should be rejected.
        """
        config_dict = {
            "cluster_name": "test-cluster",
            "instance_groups": [
                {
                    "instance_group_name": "controller",
                    "instance_type": "ml.t3.medium",
                    "instance_count": -1,  # Invalid
                    "execution_role": "arn:aws:iam::123456789012:role/TestRole",
                    "lifecycle_config": {
                        "source_s3_uri": "s3://my-bucket/scripts",
                        "on_create": "setup.sh",
                    },
                }
            ],
            "vpc_config": {
                "subnets": ["subnet-12345678"],
                "security_group_ids": ["sg-12345678"],
            },
            "slurm_config": {
                "controller_group": "controller",
                "worker_groups": [
                    {"instance_group_name": "controller", "partition_name": "default"}
                ],
            },
        }
        
        validator = SlurmClusterValidator()
        result = validator.validate_config(config_dict=config_dict)
        
        assert not result.is_valid


# =============================================================================
# Property 3: ARN format validation
# Feature: hyperpod-slurm-cluster-crud, Property 3: ARN format validation
# Validates: Requirements 9.2
# =============================================================================


class TestArnFormatValidation:
    """
    Property 3: ARN format validation
    
    *For any* string input to the ARN validator, the validator SHALL return
    true if and only if the string matches the pattern
    `arn:aws:iam::\d{12}:role/.+`.
    
    **Validates: Requirements 9.2**
    """

    @given(arn=valid_arn_strings())
    @settings(max_examples=100)
    def test_valid_arns_accepted(self, arn: str):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 3: ARN format validation
        **Validates: Requirements 9.2**
        
        Valid ARN strings should be accepted by the validator.
        """
        assert validate_arn_format(arn) is True
        
        # Also verify via validator class
        validator = SlurmClusterValidator()
        assert validator.validate_arn_format(arn) is True

    @given(arn=invalid_arn_strings())
    @settings(max_examples=100)
    def test_invalid_arns_rejected(self, arn: str):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 3: ARN format validation
        **Validates: Requirements 9.2**
        
        Invalid ARN strings should be rejected by the validator.
        """
        # Skip strings that accidentally match the pattern
        assume(not IAM_ROLE_ARN_PATTERN.match(arn))
        
        assert validate_arn_format(arn) is False
        
        # Also verify via validator class
        validator = SlurmClusterValidator()
        assert validator.validate_arn_format(arn) is False

    @given(text=st.text(min_size=0, max_size=200))
    @settings(max_examples=100)
    def test_arn_validation_consistency(self, text: str):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 3: ARN format validation
        **Validates: Requirements 9.2**
        
        ARN validation should be consistent with regex pattern.
        """
        expected = bool(IAM_ROLE_ARN_PATTERN.match(text))
        actual = validate_arn_format(text)
        
        assert actual == expected, \
            f"Validation mismatch for '{text}': expected {expected}, got {actual}"


# =============================================================================
# Property 4: S3 URI format validation
# Feature: hyperpod-slurm-cluster-crud, Property 4: S3 URI format validation
# Validates: Requirements 9.3
# =============================================================================


class TestS3UriFormatValidation:
    """
    Property 4: S3 URI format validation
    
    *For any* string input to the S3 URI validator, the validator SHALL return
    true if and only if the string matches the pattern
    `s3://[a-z0-9][a-z0-9.-]*[a-z0-9]/.+`.
    
    **Validates: Requirements 9.3**
    """

    @given(uri=valid_s3_uri_strings())
    @settings(max_examples=100)
    def test_valid_s3_uris_accepted(self, uri: str):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 4: S3 URI format validation
        **Validates: Requirements 9.3**
        
        Valid S3 URI strings should be accepted by the validator.
        """
        # Only test if it actually matches the pattern
        assume(S3_URI_PATTERN.match(uri))
        
        assert validate_s3_uri_format(uri) is True
        
        # Also verify via validator class
        validator = SlurmClusterValidator()
        assert validator.validate_s3_uri_format(uri) is True

    @given(uri=invalid_s3_uri_strings())
    @settings(max_examples=100)
    def test_invalid_s3_uris_rejected(self, uri: str):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 4: S3 URI format validation
        **Validates: Requirements 9.3**
        
        Invalid S3 URI strings should be rejected by the validator.
        """
        # Skip strings that accidentally match the pattern
        assume(not S3_URI_PATTERN.match(uri))
        
        assert validate_s3_uri_format(uri) is False
        
        # Also verify via validator class
        validator = SlurmClusterValidator()
        assert validator.validate_s3_uri_format(uri) is False

    @given(text=st.text(min_size=0, max_size=200))
    @settings(max_examples=100)
    def test_s3_uri_validation_consistency(self, text: str):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 4: S3 URI format validation
        **Validates: Requirements 9.3**
        
        S3 URI validation should be consistent with regex pattern.
        """
        expected = bool(S3_URI_PATTERN.match(text))
        actual = validate_s3_uri_format(text)
        
        assert actual == expected, \
            f"Validation mismatch for '{text}': expected {expected}, got {actual}"


# =============================================================================
# Property 12: Validation error specificity
# Feature: hyperpod-slurm-cluster-crud, Property 12: Validation error specificity
# Validates: Requirements 9.5
# =============================================================================


class TestValidationErrorSpecificity:
    """
    Property 12: Validation error specificity
    
    *For any* validation failure, the error message SHALL include the name
    of the field that failed validation.
    
    **Validates: Requirements 9.5**
    """

    def test_cluster_name_error_includes_field_name(self):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 12: Validation error specificity
        **Validates: Requirements 9.5**
        
        Cluster name validation error should include field name.
        """
        config_dict = {
            "cluster_name": "-invalid",
            "instance_groups": [
                {
                    "instance_group_name": "controller",
                    "instance_type": "ml.t3.medium",
                    "instance_count": 1,
                    "execution_role": "arn:aws:iam::123456789012:role/TestRole",
                    "lifecycle_config": {
                        "source_s3_uri": "s3://my-bucket/scripts",
                        "on_create": "setup.sh",
                    },
                }
            ],
            "vpc_config": {
                "subnets": ["subnet-12345678"],
                "security_group_ids": ["sg-12345678"],
            },
            "slurm_config": {
                "controller_group": "controller",
                "worker_groups": [
                    {"instance_group_name": "controller", "partition_name": "default"}
                ],
            },
        }
        
        validator = SlurmClusterValidator()
        result = validator.validate_config(config_dict=config_dict)
        
        assert not result.is_valid
        error_fields = [e.field for e in result.errors]
        assert any("cluster_name" in f for f in error_fields), \
            f"Error should mention cluster_name, got fields: {error_fields}"

    def test_execution_role_error_includes_field_name(self):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 12: Validation error specificity
        **Validates: Requirements 9.5**
        
        Execution role validation error should include field name.
        """
        config_dict = {
            "cluster_name": "test-cluster",
            "instance_groups": [
                {
                    "instance_group_name": "controller",
                    "instance_type": "ml.t3.medium",
                    "instance_count": 1,
                    "execution_role": "invalid-arn",
                    "lifecycle_config": {
                        "source_s3_uri": "s3://my-bucket/scripts",
                        "on_create": "setup.sh",
                    },
                }
            ],
            "vpc_config": {
                "subnets": ["subnet-12345678"],
                "security_group_ids": ["sg-12345678"],
            },
            "slurm_config": {
                "controller_group": "controller",
                "worker_groups": [
                    {"instance_group_name": "controller", "partition_name": "default"}
                ],
            },
        }
        
        validator = SlurmClusterValidator()
        result = validator.validate_config(config_dict=config_dict)
        
        assert not result.is_valid
        error_fields = [e.field for e in result.errors]
        assert any("execution_role" in f for f in error_fields), \
            f"Error should mention execution_role, got fields: {error_fields}"

    def test_s3_uri_error_includes_field_name(self):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 12: Validation error specificity
        **Validates: Requirements 9.5**
        
        S3 URI validation error should include field name.
        """
        config_dict = {
            "cluster_name": "test-cluster",
            "instance_groups": [
                {
                    "instance_group_name": "controller",
                    "instance_type": "ml.t3.medium",
                    "instance_count": 1,
                    "execution_role": "arn:aws:iam::123456789012:role/TestRole",
                    "lifecycle_config": {
                        "source_s3_uri": "invalid-uri",
                        "on_create": "setup.sh",
                    },
                }
            ],
            "vpc_config": {
                "subnets": ["subnet-12345678"],
                "security_group_ids": ["sg-12345678"],
            },
            "slurm_config": {
                "controller_group": "controller",
                "worker_groups": [
                    {"instance_group_name": "controller", "partition_name": "default"}
                ],
            },
        }
        
        validator = SlurmClusterValidator()
        result = validator.validate_config(config_dict=config_dict)
        
        assert not result.is_valid
        error_fields = [e.field for e in result.errors]
        assert any("source_s3_uri" in f or "s3" in f.lower() for f in error_fields), \
            f"Error should mention source_s3_uri, got fields: {error_fields}"

    def test_subnet_error_includes_field_name(self):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 12: Validation error specificity
        **Validates: Requirements 9.5**
        
        Subnet validation error should include field name.
        """
        config_dict = {
            "cluster_name": "test-cluster",
            "instance_groups": [
                {
                    "instance_group_name": "controller",
                    "instance_type": "ml.t3.medium",
                    "instance_count": 1,
                    "execution_role": "arn:aws:iam::123456789012:role/TestRole",
                    "lifecycle_config": {
                        "source_s3_uri": "s3://my-bucket/scripts",
                        "on_create": "setup.sh",
                    },
                }
            ],
            "vpc_config": {
                "subnets": ["invalid-subnet"],
                "security_group_ids": ["sg-12345678"],
            },
            "slurm_config": {
                "controller_group": "controller",
                "worker_groups": [
                    {"instance_group_name": "controller", "partition_name": "default"}
                ],
            },
        }
        
        validator = SlurmClusterValidator()
        result = validator.validate_config(config_dict=config_dict)
        
        assert not result.is_valid
        error_fields = [e.field for e in result.errors]
        assert any("subnet" in f.lower() for f in error_fields), \
            f"Error should mention subnet, got fields: {error_fields}"

    def test_security_group_error_includes_field_name(self):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 12: Validation error specificity
        **Validates: Requirements 9.5**
        
        Security group validation error should include field name.
        """
        config_dict = {
            "cluster_name": "test-cluster",
            "instance_groups": [
                {
                    "instance_group_name": "controller",
                    "instance_type": "ml.t3.medium",
                    "instance_count": 1,
                    "execution_role": "arn:aws:iam::123456789012:role/TestRole",
                    "lifecycle_config": {
                        "source_s3_uri": "s3://my-bucket/scripts",
                        "on_create": "setup.sh",
                    },
                }
            ],
            "vpc_config": {
                "subnets": ["subnet-12345678"],
                "security_group_ids": ["invalid-sg"],
            },
            "slurm_config": {
                "controller_group": "controller",
                "worker_groups": [
                    {"instance_group_name": "controller", "partition_name": "default"}
                ],
            },
        }
        
        validator = SlurmClusterValidator()
        result = validator.validate_config(config_dict=config_dict)
        
        assert not result.is_valid
        error_fields = [e.field for e in result.errors]
        assert any("security_group" in f.lower() for f in error_fields), \
            f"Error should mention security_group, got fields: {error_fields}"

    def test_controller_group_reference_error_includes_field_name(self):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 12: Validation error specificity
        **Validates: Requirements 9.5**
        
        Controller group reference error should include field name.
        """
        config_dict = {
            "cluster_name": "test-cluster",
            "instance_groups": [
                {
                    "instance_group_name": "controller",
                    "instance_type": "ml.t3.medium",
                    "instance_count": 1,
                    "execution_role": "arn:aws:iam::123456789012:role/TestRole",
                    "lifecycle_config": {
                        "source_s3_uri": "s3://my-bucket/scripts",
                        "on_create": "setup.sh",
                    },
                }
            ],
            "vpc_config": {
                "subnets": ["subnet-12345678"],
                "security_group_ids": ["sg-12345678"],
            },
            "slurm_config": {
                "controller_group": "nonexistent-group",  # Invalid reference
                "worker_groups": [
                    {"instance_group_name": "controller", "partition_name": "default"}
                ],
            },
        }
        
        validator = SlurmClusterValidator()
        result = validator.validate_config(config_dict=config_dict)
        
        assert not result.is_valid
        # Check that either the field name or the error message mentions controller_group
        error_fields = [e.field for e in result.errors]
        error_messages = [e.message for e in result.errors]
        has_controller_group_in_field = any("controller_group" in f for f in error_fields if f)
        has_controller_group_in_message = any("controller_group" in m for m in error_messages)
        assert has_controller_group_in_field or has_controller_group_in_message, \
            f"Error should mention controller_group, got fields: {error_fields}, messages: {error_messages}"

    @given(config=slurm_cluster_config_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_valid_config_produces_no_errors(self, config: SlurmClusterConfig):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 12: Validation error specificity
        **Validates: Requirements 9.5**
        
        Valid configurations should produce no validation errors.
        """
        validator = SlurmClusterValidator()
        result = validator.validate_config(config=config)
        
        assert result.is_valid, \
            f"Valid config should pass validation, got errors: {result.errors}"
