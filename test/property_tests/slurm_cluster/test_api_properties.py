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
Property-based tests for Slurm cluster API interactions.

These tests use Hypothesis to verify correctness properties for
API error handling, output formatting, and cluster state management.

Properties tested:
- Property 5: API error propagation (Requirements 1.3, 5.2)
- Property 6: Describe output completeness (Requirements 2.1)
- Property 7: JSON output validity (Requirements 2.2, 6.3)
- Property 8: Transitional state rejection (Requirements 3.3, 11.4)
- Property 9: Node list completeness (Requirements 6.1)
- Property 10: Orchestrator type detection (Requirements 10.1, 10.2)
- Property 11: Retry behavior on rate limiting (Requirements 11.3)
"""

import json
import string
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, call

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st
from botocore.exceptions import ClientError

from sagemaker.hyperpod.cli.slurm_cluster_utils import (
    ClusterInTransitionalStateError,
    ClusterNotFoundError,
    create_slurm_cluster,
    delete_slurm_cluster,
    describe_slurm_cluster,
    get_cluster_orchestrator_type,
    list_cluster_nodes,
    list_slurm_clusters,
    update_cluster_software,
    update_slurm_cluster,
    is_cluster_in_transitional_state,
    TRANSITIONAL_STATES,
    CLUSTER_STATUS_IN_SERVICE,
    CLUSTER_STATUS_CREATING,
    CLUSTER_STATUS_UPDATING,
    CLUSTER_STATUS_DELETING,
    CLUSTER_STATUS_FAILED,
    CLUSTER_STATUS_ROLLING_BACK,
    _retry_with_backoff,
    _is_retryable_error,
)
from sagemaker.hyperpod.cli.slurm_cluster_config import (
    SlurmClusterConfig,
    InstanceGroupConfig,
    LifecycleConfig,
    VpcConfig,
    SlurmConfig,
    WorkerGroupConfig,
    NodeRecovery,
)


# =============================================================================
# Hypothesis Strategies for generating test data
# =============================================================================


@st.composite
def error_code_strategy(draw) -> str:
    """Generate AWS error codes."""
    return draw(st.sampled_from([
        "ValidationException",
        "ResourceNotFoundException",
        "AccessDeniedException",
        "ThrottlingException",
        "ServiceUnavailable",
        "InternalServerError",
        "ConflictException",
        "ResourceLimitExceeded",
    ]))


@st.composite
def error_message_strategy(draw) -> str:
    """Generate error messages."""
    return draw(st.text(
        alphabet=string.ascii_letters + string.digits + " .-_:",
        min_size=10,
        max_size=200
    ))


@st.composite
def client_error_strategy(draw) -> ClientError:
    """Generate ClientError exceptions."""
    error_code = draw(error_code_strategy())
    error_message = draw(error_message_strategy())
    
    error_response = {
        "Error": {
            "Code": error_code,
            "Message": error_message,
        }
    }
    return ClientError(error_response, "TestOperation")


@st.composite
def cluster_status_strategy(draw) -> str:
    """Generate cluster status values."""
    return draw(st.sampled_from([
        "InService",
        "Creating",
        "Updating",
        "Deleting",
        "Failed",
        "RollingBack",
        "SystemUpdating",
    ]))


@st.composite
def transitional_status_strategy(draw) -> str:
    """Generate transitional cluster status values."""
    return draw(st.sampled_from(list(TRANSITIONAL_STATES)))


@st.composite
def non_transitional_status_strategy(draw) -> str:
    """Generate non-transitional cluster status values."""
    return draw(st.sampled_from(["InService", "Failed"]))


@st.composite
def cluster_name_strategy(draw) -> str:
    """Generate valid cluster names."""
    first_char = draw(st.sampled_from(string.ascii_letters + string.digits))
    middle = draw(st.text(
        alphabet=string.ascii_letters + string.digits + "-",
        min_size=0,
        max_size=30
    ))
    # Ensure no consecutive hyphens and doesn't end with hyphen
    middle = middle.replace("--", "-a-")
    if middle and middle[-1] == "-":
        middle = middle[:-1] + "a"
    return first_char + middle


@st.composite
def instance_group_response_strategy(draw) -> Dict[str, Any]:
    """Generate instance group response data."""
    return {
        "InstanceGroupName": draw(st.text(
            alphabet=string.ascii_letters + string.digits + "-",
            min_size=1,
            max_size=30
        )),
        "InstanceType": draw(st.sampled_from([
            "ml.t3.medium",
            "ml.m5.xlarge",
            "ml.p4d.24xlarge",
        ])),
        "CurrentCount": draw(st.integers(min_value=0, max_value=100)),
        "TargetCount": draw(st.integers(min_value=0, max_value=100)),
        "ExecutionRole": f"arn:aws:iam::{draw(st.text(alphabet=string.digits, min_size=12, max_size=12))}:role/TestRole",
    }


@st.composite
def vpc_config_response_strategy(draw) -> Dict[str, Any]:
    """Generate VPC config response data."""
    num_subnets = draw(st.integers(min_value=1, max_value=3))
    num_sgs = draw(st.integers(min_value=1, max_value=3))
    
    return {
        "Subnets": [f"subnet-{draw(st.text(alphabet='0123456789abcdef', min_size=8, max_size=8))}" 
                   for _ in range(num_subnets)],
        "SecurityGroupIds": [f"sg-{draw(st.text(alphabet='0123456789abcdef', min_size=8, max_size=8))}" 
                            for _ in range(num_sgs)],
    }


@st.composite
def cluster_details_strategy(draw) -> Dict[str, Any]:
    """Generate cluster details response data."""
    cluster_name = draw(cluster_name_strategy())
    status = draw(cluster_status_strategy())
    
    num_instance_groups = draw(st.integers(min_value=1, max_value=3))
    instance_groups = [draw(instance_group_response_strategy()) 
                      for _ in range(num_instance_groups)]
    
    vpc_config = draw(vpc_config_response_strategy())
    
    # Generate tags
    num_tags = draw(st.integers(min_value=0, max_value=5))
    tags = [
        {"Key": f"tag-{i}", "Value": f"value-{i}"}
        for i in range(num_tags)
    ]
    
    return {
        "ClusterArn": f"arn:aws:sagemaker:us-west-2:123456789012:cluster/{cluster_name}",
        "ClusterName": cluster_name,
        "ClusterStatus": status,
        "CreationTime": datetime.now(),
        "InstanceGroups": instance_groups,
        "VpcConfig": vpc_config,
        "Tags": tags,
        "NodeRecovery": draw(st.sampled_from(["Automatic", "None"])),
    }


@st.composite
def node_summary_strategy(draw) -> Dict[str, Any]:
    """Generate cluster node summary data."""
    return {
        "InstanceId": f"i-{draw(st.text(alphabet='0123456789abcdef', min_size=17, max_size=17))}",
        "InstanceType": draw(st.sampled_from([
            "ml.t3.medium",
            "ml.m5.xlarge",
            "ml.p4d.24xlarge",
        ])),
        "InstanceStatus": {
            "Status": draw(st.sampled_from(["Running", "Pending", "Stopping", "Stopped"])),
            "Message": draw(st.text(min_size=0, max_size=50)),
        },
        "LaunchTime": datetime.now(),
    }


@st.composite
def slurm_cluster_config_strategy(draw) -> SlurmClusterConfig:
    """Generate valid SlurmClusterConfig objects for testing."""
    cluster_name = draw(cluster_name_strategy())
    assume(len(cluster_name) >= 1)
    
    return SlurmClusterConfig(
        cluster_name=cluster_name,
        instance_groups=[
            InstanceGroupConfig(
                instance_group_name="controller",
                instance_type="ml.t3.medium",
                instance_count=1,
                execution_role="arn:aws:iam::123456789012:role/TestRole",
                lifecycle_config=LifecycleConfig(
                    source_s3_uri="s3://my-bucket/scripts",
                    on_create="setup.sh",
                ),
            ),
        ],
        vpc_config=VpcConfig(
            subnets=["subnet-12345678"],
            security_group_ids=["sg-12345678"],
        ),
        slurm_config=SlurmConfig(
            controller_group="controller",
            worker_groups=[
                WorkerGroupConfig(
                    instance_group_name="controller",
                    partition_name="default",
                ),
            ],
        ),
    )


# =============================================================================
# Property 5: API error propagation
# Feature: hyperpod-slurm-cluster-crud, Property 5: API error propagation
# Validates: Requirements 1.3, 5.2
# =============================================================================


class TestApiErrorPropagation:
    """
    Property 5: API error propagation
    
    *For any* API error response, the CLI SHALL include the error message
    from the API response in its output.
    
    **Validates: Requirements 1.3, 5.2**
    """

    @given(error=client_error_strategy())
    @settings(max_examples=100, deadline=None)
    def test_create_cluster_propagates_error_message(self, error: ClientError):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 5: API error propagation
        **Validates: Requirements 1.3, 5.2**
        
        Create cluster should propagate API error messages.
        """
        mock_client = MagicMock()
        mock_client.create_cluster.side_effect = error
        
        config = SlurmClusterConfig(
            cluster_name="test-cluster",
            instance_groups=[
                InstanceGroupConfig(
                    instance_group_name="controller",
                    instance_type="ml.t3.medium",
                    instance_count=1,
                    execution_role="arn:aws:iam::123456789012:role/TestRole",
                    lifecycle_config=LifecycleConfig(
                        source_s3_uri="s3://my-bucket/scripts",
                        on_create="setup.sh",
                    ),
                ),
            ],
            vpc_config=VpcConfig(
                subnets=["subnet-12345678"],
                security_group_ids=["sg-12345678"],
            ),
            slurm_config=SlurmConfig(
                controller_group="controller",
                worker_groups=[
                    WorkerGroupConfig(
                        instance_group_name="controller",
                        partition_name="default",
                    ),
                ],
            ),
        )
        
        # Skip retryable errors as they will be retried
        error_code = error.response.get("Error", {}).get("Code", "")
        if error_code in ["ThrottlingException", "ServiceUnavailable", "InternalServerError"]:
            # These will be retried, so we need to set up multiple failures
            mock_client.create_cluster.side_effect = [error] * 4
        
        # Patch time.sleep to avoid actual delays during retry
        with patch('time.sleep'):
            with pytest.raises(ClientError) as exc_info:
                create_slurm_cluster(mock_client, config)
        
        # Verify error message is preserved
        caught_error = exc_info.value
        original_message = error.response.get("Error", {}).get("Message", "")
        caught_message = caught_error.response.get("Error", {}).get("Message", "")
        
        assert original_message == caught_message, \
            f"Error message should be preserved: expected '{original_message}', got '{caught_message}'"

    @given(error_code=error_code_strategy(), error_message=error_message_strategy())
    @settings(max_examples=100, deadline=None)
    def test_describe_cluster_propagates_error_message(
        self, error_code: str, error_message: str
    ):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 5: API error propagation
        **Validates: Requirements 1.3, 5.2**
        
        Describe cluster should propagate API error messages.
        """
        mock_client = MagicMock()
        error_response = {
            "Error": {
                "Code": error_code,
                "Message": error_message,
            }
        }
        error = ClientError(error_response, "DescribeCluster")
        
        # For retryable errors, set up multiple failures
        if error_code in ["ThrottlingException", "ServiceUnavailable", "InternalServerError"]:
            mock_client.describe_cluster.side_effect = [error] * 4
        else:
            mock_client.describe_cluster.side_effect = error
        
        # Patch time.sleep to avoid actual delays during retry
        with patch('time.sleep'):
            # ResourceNotFoundException is converted to ClusterNotFoundError
            if error_code == "ResourceNotFoundException":
                with pytest.raises(ClusterNotFoundError):
                    describe_slurm_cluster(mock_client, "test-cluster")
            else:
                with pytest.raises(ClientError) as exc_info:
                    describe_slurm_cluster(mock_client, "test-cluster")
                
                caught_error = exc_info.value
                caught_message = caught_error.response.get("Error", {}).get("Message", "")
                
                assert error_message == caught_message, \
                    f"Error message should be preserved: expected '{error_message}', got '{caught_message}'"

    @given(error_code=error_code_strategy(), error_message=error_message_strategy())
    @settings(max_examples=100, deadline=None)
    def test_update_software_propagates_error_message(
        self, error_code: str, error_message: str
    ):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 5: API error propagation
        **Validates: Requirements 1.3, 5.2**
        
        Update software should propagate API error messages.
        """
        mock_client = MagicMock()
        error_response = {
            "Error": {
                "Code": error_code,
                "Message": error_message,
            }
        }
        error = ClientError(error_response, "UpdateClusterSoftware")
        
        # For retryable errors, set up multiple failures
        if error_code in ["ThrottlingException", "ServiceUnavailable", "InternalServerError"]:
            mock_client.update_cluster_software.side_effect = [error] * 4
        else:
            mock_client.update_cluster_software.side_effect = error
        
        # Patch time.sleep to avoid actual delays during retry
        with patch('time.sleep'):
            # ResourceNotFoundException is converted to ClusterNotFoundError
            if error_code == "ResourceNotFoundException":
                with pytest.raises(ClusterNotFoundError):
                    update_cluster_software(mock_client, "test-cluster")
            else:
                with pytest.raises(ClientError) as exc_info:
                    update_cluster_software(mock_client, "test-cluster")
                
                caught_error = exc_info.value
                caught_message = caught_error.response.get("Error", {}).get("Message", "")
                
                assert error_message == caught_message, \
                    f"Error message should be preserved"


# =============================================================================
# Property 6: Describe output completeness
# Feature: hyperpod-slurm-cluster-crud, Property 6: Describe output completeness
# Validates: Requirements 2.1
# =============================================================================


class TestDescribeOutputCompleteness:
    """
    Property 6: Describe output completeness
    
    *For any* cluster details returned by the DescribeCluster API, the describe
    command output SHALL contain the cluster status, instance groups, VPC config,
    and tags fields.
    
    **Validates: Requirements 2.1**
    """

    @given(cluster_details=cluster_details_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_describe_returns_all_required_fields(
        self, cluster_details: Dict[str, Any]
    ):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 6: Describe output completeness
        **Validates: Requirements 2.1**
        
        Describe should return all required fields from API response.
        """
        mock_client = MagicMock()
        mock_client.describe_cluster.return_value = cluster_details
        
        result = describe_slurm_cluster(mock_client, cluster_details["ClusterName"])
        
        # Verify all required fields are present
        assert "ClusterStatus" in result, "ClusterStatus should be in response"
        assert "InstanceGroups" in result, "InstanceGroups should be in response"
        assert "VpcConfig" in result, "VpcConfig should be in response"
        assert "Tags" in result, "Tags should be in response"
        
        # Verify values match
        assert result["ClusterStatus"] == cluster_details["ClusterStatus"]
        assert result["InstanceGroups"] == cluster_details["InstanceGroups"]
        assert result["VpcConfig"] == cluster_details["VpcConfig"]
        assert result["Tags"] == cluster_details["Tags"]

    @given(cluster_details=cluster_details_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_describe_preserves_instance_group_details(
        self, cluster_details: Dict[str, Any]
    ):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 6: Describe output completeness
        **Validates: Requirements 2.1**
        
        Describe should preserve all instance group details.
        """
        mock_client = MagicMock()
        mock_client.describe_cluster.return_value = cluster_details
        
        result = describe_slurm_cluster(mock_client, cluster_details["ClusterName"])
        
        # Verify instance groups are complete
        for i, ig in enumerate(result["InstanceGroups"]):
            original_ig = cluster_details["InstanceGroups"][i]
            assert ig["InstanceGroupName"] == original_ig["InstanceGroupName"]
            assert ig["InstanceType"] == original_ig["InstanceType"]

    @given(cluster_details=cluster_details_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_describe_preserves_vpc_config(
        self, cluster_details: Dict[str, Any]
    ):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 6: Describe output completeness
        **Validates: Requirements 2.1**
        
        Describe should preserve VPC configuration.
        """
        mock_client = MagicMock()
        mock_client.describe_cluster.return_value = cluster_details
        
        result = describe_slurm_cluster(mock_client, cluster_details["ClusterName"])
        
        # Verify VPC config is complete
        assert result["VpcConfig"]["Subnets"] == cluster_details["VpcConfig"]["Subnets"]
        assert result["VpcConfig"]["SecurityGroupIds"] == cluster_details["VpcConfig"]["SecurityGroupIds"]


# =============================================================================
# Property 7: JSON output validity
# Feature: hyperpod-slurm-cluster-crud, Property 7: JSON output validity
# Validates: Requirements 2.2, 6.3
# =============================================================================


class TestJsonOutputValidity:
    """
    Property 7: JSON output validity
    
    *For any* command with `--output json` flag, the output SHALL be valid
    JSON that can be parsed without errors.
    
    **Validates: Requirements 2.2, 6.3**
    """

    @given(cluster_details=cluster_details_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_cluster_details_serializable_to_json(
        self, cluster_details: Dict[str, Any]
    ):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 7: JSON output validity
        **Validates: Requirements 2.2, 6.3**
        
        Cluster details should be serializable to valid JSON.
        """
        mock_client = MagicMock()
        mock_client.describe_cluster.return_value = cluster_details
        
        result = describe_slurm_cluster(mock_client, cluster_details["ClusterName"])
        
        # Serialize to JSON
        json_str = json.dumps(result, default=str)
        
        # Verify it can be parsed back
        parsed = json.loads(json_str)
        
        assert isinstance(parsed, dict)
        assert "ClusterStatus" in parsed
        assert "InstanceGroups" in parsed

    @given(nodes=st.lists(node_summary_strategy(), min_size=0, max_size=10))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_node_list_serializable_to_json(self, nodes: List[Dict[str, Any]]):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 7: JSON output validity
        **Validates: Requirements 2.2, 6.3**
        
        Node list should be serializable to valid JSON.
        """
        mock_client = MagicMock()
        mock_client.list_cluster_nodes.return_value = {
            "ClusterNodeSummaries": nodes,
        }
        
        result = list_cluster_nodes(mock_client, "test-cluster")
        
        # Serialize to JSON
        json_str = json.dumps(result, default=str)
        
        # Verify it can be parsed back
        parsed = json.loads(json_str)
        
        assert isinstance(parsed, dict)
        assert "ClusterNodeSummaries" in parsed
        assert len(parsed["ClusterNodeSummaries"]) == len(nodes)

    @given(cluster_details=cluster_details_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_json_output_is_valid_json(self, cluster_details: Dict[str, Any]):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 7: JSON output validity
        **Validates: Requirements 2.2, 6.3**
        
        JSON output should be parseable without errors.
        """
        # Serialize with default=str to handle datetime
        json_str = json.dumps(cluster_details, default=str)
        
        # Should not raise
        parsed = json.loads(json_str)
        
        # Verify structure is preserved
        assert parsed["ClusterName"] == cluster_details["ClusterName"]
        assert parsed["ClusterStatus"] == cluster_details["ClusterStatus"]


# =============================================================================
# Property 8: Transitional state rejection
# Feature: hyperpod-slurm-cluster-crud, Property 8: Transitional state rejection
# Validates: Requirements 3.3, 11.4
# =============================================================================


class TestTransitionalStateRejection:
    """
    Property 8: Transitional state rejection
    
    *For any* cluster in a transitional state (Creating, Updating, Deleting,
    RollingBack), update operations SHALL return an error indicating the
    cluster cannot be updated.
    
    **Validates: Requirements 3.3, 11.4**
    """

    @given(status=transitional_status_strategy())
    @settings(max_examples=100)
    def test_update_rejected_in_transitional_state(self, status: str):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 8: Transitional state rejection
        **Validates: Requirements 3.3, 11.4**
        
        Updates should be rejected when cluster is in transitional state.
        """
        mock_client = MagicMock()
        mock_client.describe_cluster.return_value = {
            "ClusterName": "test-cluster",
            "ClusterStatus": status,
        }
        
        with pytest.raises(ClusterInTransitionalStateError) as exc_info:
            update_slurm_cluster(
                mock_client,
                "test-cluster",
                node_recovery="None",
            )
        
        # Verify error message includes the status
        assert status in str(exc_info.value), \
            f"Error message should include status '{status}'"
        assert "cannot be updated" in str(exc_info.value).lower(), \
            "Error message should indicate cluster cannot be updated"

    @given(status=non_transitional_status_strategy())
    @settings(max_examples=100)
    def test_update_allowed_in_stable_state(self, status: str):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 8: Transitional state rejection
        **Validates: Requirements 3.3, 11.4**
        
        Updates should be allowed when cluster is in stable state.
        """
        mock_client = MagicMock()
        mock_client.describe_cluster.return_value = {
            "ClusterName": "test-cluster",
            "ClusterStatus": status,
        }
        mock_client.update_cluster.return_value = {
            "ClusterArn": "arn:aws:sagemaker:us-west-2:123456789012:cluster/test-cluster"
        }
        
        # Should not raise
        result = update_slurm_cluster(
            mock_client,
            "test-cluster",
            node_recovery="None",
        )
        
        assert "ClusterArn" in result

    @given(status=transitional_status_strategy())
    @settings(max_examples=100)
    def test_is_transitional_state_returns_true(self, status: str):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 8: Transitional state rejection
        **Validates: Requirements 3.3, 11.4**
        
        is_cluster_in_transitional_state should return True for transitional states.
        """
        assert is_cluster_in_transitional_state(status) is True

    @given(status=non_transitional_status_strategy())
    @settings(max_examples=100)
    def test_is_transitional_state_returns_false(self, status: str):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 8: Transitional state rejection
        **Validates: Requirements 3.3, 11.4**
        
        is_cluster_in_transitional_state should return False for stable states.
        """
        assert is_cluster_in_transitional_state(status) is False

    def test_all_transitional_states_covered(self):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 8: Transitional state rejection
        **Validates: Requirements 3.3, 11.4**
        
        All expected transitional states should be in TRANSITIONAL_STATES.
        """
        expected_transitional = {
            "Creating",
            "Updating",
            "Deleting",
            "RollingBack",
            "SystemUpdating",
        }
        
        assert expected_transitional.issubset(TRANSITIONAL_STATES), \
            f"Missing transitional states: {expected_transitional - TRANSITIONAL_STATES}"


# =============================================================================
# Property 9: Node list completeness
# Feature: hyperpod-slurm-cluster-crud, Property 9: Node list completeness
# Validates: Requirements 6.1
# =============================================================================


class TestNodeListCompleteness:
    """
    Property 9: Node list completeness
    
    *For any* node returned by the ListClusterNodes API, the list-nodes command
    output SHALL contain the node ID, instance type, status, and health fields.
    
    **Validates: Requirements 6.1**
    """

    @given(nodes=st.lists(node_summary_strategy(), min_size=1, max_size=10))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_node_list_contains_required_fields(self, nodes: List[Dict[str, Any]]):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 9: Node list completeness
        **Validates: Requirements 6.1**
        
        Node list should contain all required fields for each node.
        """
        mock_client = MagicMock()
        mock_client.list_cluster_nodes.return_value = {
            "ClusterNodeSummaries": nodes,
        }
        
        result = list_cluster_nodes(mock_client, "test-cluster")
        
        returned_nodes = result.get("ClusterNodeSummaries", [])
        assert len(returned_nodes) == len(nodes)
        
        for i, node in enumerate(returned_nodes):
            # Verify required fields are present
            assert "InstanceId" in node, f"Node {i} should have InstanceId"
            assert "InstanceType" in node, f"Node {i} should have InstanceType"
            assert "InstanceStatus" in node, f"Node {i} should have InstanceStatus"
            
            # Verify status has required subfields
            status = node["InstanceStatus"]
            assert "Status" in status, f"Node {i} InstanceStatus should have Status"

    @given(nodes=st.lists(node_summary_strategy(), min_size=1, max_size=10))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_node_list_preserves_all_data(self, nodes: List[Dict[str, Any]]):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 9: Node list completeness
        **Validates: Requirements 6.1**
        
        Node list should preserve all data from API response.
        """
        mock_client = MagicMock()
        mock_client.list_cluster_nodes.return_value = {
            "ClusterNodeSummaries": nodes,
        }
        
        result = list_cluster_nodes(mock_client, "test-cluster")
        
        returned_nodes = result.get("ClusterNodeSummaries", [])
        
        for i, node in enumerate(returned_nodes):
            original = nodes[i]
            assert node["InstanceId"] == original["InstanceId"]
            assert node["InstanceType"] == original["InstanceType"]
            assert node["InstanceStatus"]["Status"] == original["InstanceStatus"]["Status"]

    def test_empty_node_list_handled(self):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 9: Node list completeness
        **Validates: Requirements 6.1**
        
        Empty node list should be handled gracefully.
        """
        mock_client = MagicMock()
        mock_client.list_cluster_nodes.return_value = {
            "ClusterNodeSummaries": [],
        }
        
        result = list_cluster_nodes(mock_client, "test-cluster")
        
        assert result.get("ClusterNodeSummaries") == []


# =============================================================================
# Property 10: Orchestrator type detection
# Feature: hyperpod-slurm-cluster-crud, Property 10: Orchestrator type detection
# Validates: Requirements 10.1, 10.2
# =============================================================================


class TestOrchestratorTypeDetection:
    """
    Property 10: Orchestrator type detection
    
    *For any* cluster, the orchestrator type detection function SHALL return
    "Slurm" if the cluster has no Orchestrator.Eks field, and "EKS" if it does.
    
    **Validates: Requirements 10.1, 10.2**
    """

    @given(cluster_details=cluster_details_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_slurm_cluster_detected_without_eks(self, cluster_details: Dict[str, Any]):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 10: Orchestrator type detection
        **Validates: Requirements 10.1, 10.2**
        
        Clusters without EKS orchestrator should be detected as Slurm.
        """
        # Ensure no EKS orchestrator
        cluster_details["Orchestrator"] = {}
        
        result = get_cluster_orchestrator_type(cluster_details)
        
        assert result == "Slurm"

    @given(cluster_details=cluster_details_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_eks_cluster_detected_with_eks(self, cluster_details: Dict[str, Any]):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 10: Orchestrator type detection
        **Validates: Requirements 10.1, 10.2**
        
        Clusters with EKS orchestrator should be detected as EKS.
        """
        # Add EKS orchestrator
        cluster_details["Orchestrator"] = {
            "Eks": {
                "ClusterArn": "arn:aws:eks:us-west-2:123456789012:cluster/test"
            }
        }
        
        result = get_cluster_orchestrator_type(cluster_details)
        
        assert result == "EKS"

    def test_missing_orchestrator_field_is_slurm(self):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 10: Orchestrator type detection
        **Validates: Requirements 10.1, 10.2**
        
        Missing Orchestrator field should default to Slurm.
        """
        cluster_details = {
            "ClusterName": "test-cluster",
            "ClusterStatus": "InService",
        }
        
        result = get_cluster_orchestrator_type(cluster_details)
        
        assert result == "Slurm"

    def test_empty_orchestrator_is_slurm(self):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 10: Orchestrator type detection
        **Validates: Requirements 10.1, 10.2**
        
        Empty Orchestrator field should be Slurm.
        """
        cluster_details = {
            "ClusterName": "test-cluster",
            "ClusterStatus": "InService",
            "Orchestrator": {},
        }
        
        result = get_cluster_orchestrator_type(cluster_details)
        
        assert result == "Slurm"

    def test_orchestrator_with_none_eks_is_slurm(self):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 10: Orchestrator type detection
        **Validates: Requirements 10.1, 10.2**
        
        Orchestrator with None Eks field should be Slurm.
        """
        cluster_details = {
            "ClusterName": "test-cluster",
            "ClusterStatus": "InService",
            "Orchestrator": {"Eks": None},
        }
        
        result = get_cluster_orchestrator_type(cluster_details)
        
        assert result == "Slurm"

    @given(eks_arn=st.text(min_size=1, max_size=100))
    @settings(max_examples=100)
    def test_any_eks_arn_detected_as_eks(self, eks_arn: str):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 10: Orchestrator type detection
        **Validates: Requirements 10.1, 10.2**
        
        Any non-empty EKS configuration should be detected as EKS.
        """
        cluster_details = {
            "ClusterName": "test-cluster",
            "ClusterStatus": "InService",
            "Orchestrator": {
                "Eks": {"ClusterArn": eks_arn}
            },
        }
        
        result = get_cluster_orchestrator_type(cluster_details)
        
        assert result == "EKS"


# =============================================================================
# Property 11: Retry behavior on rate limiting
# Feature: hyperpod-slurm-cluster-crud, Property 11: Retry behavior on rate limiting
# Validates: Requirements 11.3
# =============================================================================


class TestRetryBehavior:
    """
    Property 11: Retry behavior on rate limiting
    
    *For any* API call that receives a rate limiting error, the CLI SHALL
    retry up to 3 times with exponential backoff before returning an error.
    
    **Validates: Requirements 11.3**
    """

    def test_throttling_triggers_retry(self):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 11: Retry behavior on rate limiting
        **Validates: Requirements 11.3**
        
        ThrottlingException should trigger retry.
        """
        error_response = {
            "Error": {
                "Code": "ThrottlingException",
                "Message": "Rate exceeded",
            }
        }
        error = ClientError(error_response, "TestOperation")
        
        assert _is_retryable_error(error) is True

    def test_service_unavailable_triggers_retry(self):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 11: Retry behavior on rate limiting
        **Validates: Requirements 11.3**
        
        ServiceUnavailable should trigger retry.
        """
        error_response = {
            "Error": {
                "Code": "ServiceUnavailable",
                "Message": "Service temporarily unavailable",
            }
        }
        error = ClientError(error_response, "TestOperation")
        
        assert _is_retryable_error(error) is True

    def test_internal_server_error_triggers_retry(self):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 11: Retry behavior on rate limiting
        **Validates: Requirements 11.3**
        
        InternalServerError should trigger retry.
        """
        error_response = {
            "Error": {
                "Code": "InternalServerError",
                "Message": "Internal server error",
            }
        }
        error = ClientError(error_response, "TestOperation")
        
        assert _is_retryable_error(error) is True

    def test_validation_exception_does_not_retry(self):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 11: Retry behavior on rate limiting
        **Validates: Requirements 11.3**
        
        ValidationException should not trigger retry.
        """
        error_response = {
            "Error": {
                "Code": "ValidationException",
                "Message": "Invalid parameter",
            }
        }
        error = ClientError(error_response, "TestOperation")
        
        assert _is_retryable_error(error) is False

    def test_access_denied_does_not_retry(self):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 11: Retry behavior on rate limiting
        **Validates: Requirements 11.3**
        
        AccessDeniedException should not trigger retry.
        """
        error_response = {
            "Error": {
                "Code": "AccessDeniedException",
                "Message": "Access denied",
            }
        }
        error = ClientError(error_response, "TestOperation")
        
        assert _is_retryable_error(error) is False

    def test_retry_succeeds_after_transient_failure(self):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 11: Retry behavior on rate limiting
        **Validates: Requirements 11.3**
        
        Retry should succeed after transient failures.
        """
        throttle_error = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
            "TestOperation"
        )
        
        call_count = [0]
        
        def mock_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise throttle_error
            return "success"
        
        result = _retry_with_backoff(mock_func, max_retries=3, base_delay=0.01)
        
        assert result == "success"
        assert call_count[0] == 3  # 2 failures + 1 success

    def test_retry_exhausted_raises_error(self):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 11: Retry behavior on rate limiting
        **Validates: Requirements 11.3**
        
        After max retries, error should be raised.
        """
        throttle_error = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
            "TestOperation"
        )
        
        def mock_func():
            raise throttle_error
        
        with pytest.raises(ClientError):
            _retry_with_backoff(mock_func, max_retries=3, base_delay=0.01)

    def test_describe_cluster_retries_on_throttling(self):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 11: Retry behavior on rate limiting
        **Validates: Requirements 11.3**
        
        describe_slurm_cluster should retry on throttling.
        """
        mock_client = MagicMock()
        
        throttle_error = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
            "DescribeCluster"
        )
        success_response = {
            "ClusterName": "test-cluster",
            "ClusterStatus": "InService",
        }
        
        # First two calls fail, third succeeds
        mock_client.describe_cluster.side_effect = [
            throttle_error,
            throttle_error,
            success_response,
        ]
        
        result = describe_slurm_cluster(mock_client, "test-cluster")
        
        assert result["ClusterStatus"] == "InService"
        assert mock_client.describe_cluster.call_count == 3

    @given(num_failures=st.integers(min_value=1, max_value=3))
    @settings(max_examples=10)
    def test_retry_count_matches_failures(self, num_failures: int):
        """
        Feature: hyperpod-slurm-cluster-crud, Property 11: Retry behavior on rate limiting
        **Validates: Requirements 11.3**
        
        Number of retries should match number of transient failures.
        """
        throttle_error = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
            "TestOperation"
        )
        
        call_count = [0]
        
        def mock_func():
            call_count[0] += 1
            if call_count[0] <= num_failures:
                raise throttle_error
            return "success"
        
        result = _retry_with_backoff(mock_func, max_retries=3, base_delay=0.01)
        
        assert result == "success"
        assert call_count[0] == num_failures + 1
