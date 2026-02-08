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
Integration tests for Slurm cluster CRUD lifecycle.

These tests verify the create → describe → update → delete workflow
for Slurm-orchestrated HyperPod clusters using mocked boto3 clients.

Requirements tested:
- 1.1: Create Slurm clusters via CLI
- 2.1: Describe Slurm clusters via CLI
- 3.1: Update Slurm cluster configurations via CLI
- 4.1: Delete Slurm clusters via CLI
"""

import json
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from sagemaker.hyperpod.cli.slurm_cluster_config import (
    InstanceGroupConfig,
    LifecycleConfig,
    NodeRecovery,
    SlurmClusterConfig,
    SlurmConfig,
    VpcConfig,
    WorkerGroupConfig,
)
from sagemaker.hyperpod.cli.slurm_cluster_utils import (
    ClusterInTransitionalStateError,
    ClusterNotFoundError,
    create_slurm_cluster,
    delete_slurm_cluster,
    describe_slurm_cluster,
    get_cluster_orchestrator_type,
    list_slurm_clusters,
    update_slurm_cluster,
)


class TestSlurmClusterLifecycle:
    """Integration tests for Slurm cluster CRUD lifecycle with mocked boto3."""

    @pytest.fixture
    def mock_sm_client(self):
        """Create a mock SageMaker client."""
        return MagicMock()

    @pytest.fixture
    def sample_cluster_name(self):
        """Generate a unique cluster name for testing."""
        return f"test-slurm-cluster-{str(uuid.uuid4())[:8]}"

    @pytest.fixture
    def sample_cluster_config(self, sample_cluster_name):
        """Create a sample Slurm cluster configuration."""
        return SlurmClusterConfig(
            cluster_name=sample_cluster_name,
            region="us-west-2",
            instance_groups=[
                InstanceGroupConfig(
                    instance_group_name="controller",
                    instance_type="ml.m5.xlarge",
                    instance_count=1,
                    execution_role="arn:aws:iam::123456789012:role/SageMakerRole",
                    lifecycle_config=LifecycleConfig(
                        source_s3_uri="s3://my-bucket/lifecycle-scripts",
                        on_create="setup.sh",
                    ),
                ),
                InstanceGroupConfig(
                    instance_group_name="workers",
                    instance_type="ml.p4d.24xlarge",
                    instance_count=2,
                    execution_role="arn:aws:iam::123456789012:role/SageMakerRole",
                    lifecycle_config=LifecycleConfig(
                        source_s3_uri="s3://my-bucket/lifecycle-scripts",
                        on_create="setup.sh",
                    ),
                ),
            ],
            vpc_config=VpcConfig(
                subnets=["subnet-12345678"],
                security_group_ids=["sg-12345678"],
            ),
            node_recovery=NodeRecovery.AUTOMATIC,
            slurm_config=SlurmConfig(
                controller_group="controller",
                worker_groups=[
                    WorkerGroupConfig(
                        instance_group_name="workers",
                        partition_name="compute",
                    ),
                ],
            ),
        )

    @pytest.fixture
    def sample_cluster_response(self, sample_cluster_name):
        """Create a sample cluster describe response."""
        return {
            "ClusterArn": f"arn:aws:sagemaker:us-west-2:123456789012:cluster/{sample_cluster_name}",
            "ClusterName": sample_cluster_name,
            "ClusterStatus": "InService",
            "CreationTime": datetime.now(),
            "InstanceGroups": [
                {
                    "InstanceGroupName": "controller",
                    "InstanceType": "ml.m5.xlarge",
                    "CurrentCount": 1,
                    "TargetCount": 1,
                    "ExecutionRole": "arn:aws:iam::123456789012:role/SageMakerRole",
                },
                {
                    "InstanceGroupName": "workers",
                    "InstanceType": "ml.p4d.24xlarge",
                    "CurrentCount": 2,
                    "TargetCount": 2,
                    "ExecutionRole": "arn:aws:iam::123456789012:role/SageMakerRole",
                },
            ],
            "VpcConfig": {
                "Subnets": ["subnet-12345678"],
                "SecurityGroupIds": ["sg-12345678"],
            },
            "NodeRecovery": "Automatic",
            "Tags": [],
        }

    # =========================================================================
    # Create Tests - Requirement 1.1
    # =========================================================================

    def test_create_cluster_success(
        self, mock_sm_client, sample_cluster_config, sample_cluster_name
    ):
        """Test successful cluster creation returns ARN."""
        expected_arn = f"arn:aws:sagemaker:us-west-2:123456789012:cluster/{sample_cluster_name}"
        mock_sm_client.create_cluster.return_value = {"ClusterArn": expected_arn}

        result = create_slurm_cluster(mock_sm_client, sample_cluster_config)

        assert result == expected_arn
        mock_sm_client.create_cluster.assert_called_once()

        # Verify the request structure
        call_args = mock_sm_client.create_cluster.call_args
        request = call_args.kwargs
        assert request["ClusterName"] == sample_cluster_name
        assert len(request["InstanceGroups"]) == 2
        assert "VpcConfig" in request

    def test_create_cluster_api_error(self, mock_sm_client, sample_cluster_config):
        """Test that API errors are propagated correctly."""
        error_response = {
            "Error": {
                "Code": "ValidationException",
                "Message": "Invalid instance type specified",
            }
        }
        mock_sm_client.create_cluster.side_effect = ClientError(
            error_response, "CreateCluster"
        )

        with pytest.raises(ClientError) as exc_info:
            create_slurm_cluster(mock_sm_client, sample_cluster_config)

        assert "ValidationException" in str(exc_info.value)

    # =========================================================================
    # Describe Tests - Requirement 2.1
    # =========================================================================

    def test_describe_cluster_success(
        self, mock_sm_client, sample_cluster_name, sample_cluster_response
    ):
        """Test successful cluster description returns all required fields."""
        mock_sm_client.describe_cluster.return_value = sample_cluster_response

        result = describe_slurm_cluster(mock_sm_client, sample_cluster_name)

        assert result["ClusterName"] == sample_cluster_name
        assert result["ClusterStatus"] == "InService"
        assert "InstanceGroups" in result
        assert "VpcConfig" in result
        assert "Tags" in result
        mock_sm_client.describe_cluster.assert_called_once_with(
            ClusterName=sample_cluster_name
        )

    def test_describe_cluster_not_found(self, mock_sm_client, sample_cluster_name):
        """Test that ClusterNotFoundError is raised for non-existent clusters."""
        error_response = {
            "Error": {
                "Code": "ResourceNotFoundException",
                "Message": f"Cluster {sample_cluster_name} not found",
            }
        }
        mock_sm_client.describe_cluster.side_effect = ClientError(
            error_response, "DescribeCluster"
        )

        with pytest.raises(ClusterNotFoundError) as exc_info:
            describe_slurm_cluster(mock_sm_client, sample_cluster_name)

        assert sample_cluster_name in str(exc_info.value)

    # =========================================================================
    # List Tests - Requirement 10.1
    # =========================================================================

    def test_list_clusters_filters_slurm_only(self, mock_sm_client):
        """Test that list_slurm_clusters filters to only Slurm clusters."""
        # Mock list_clusters to return both Slurm and EKS clusters
        mock_sm_client.list_clusters.return_value = {
            "ClusterSummaries": [
                {"ClusterName": "slurm-cluster-1", "ClusterStatus": "InService"},
                {"ClusterName": "eks-cluster-1", "ClusterStatus": "InService"},
            ]
        }

        # Mock describe_cluster to return different orchestrator types
        def describe_side_effect(ClusterName):
            if "slurm" in ClusterName:
                return {
                    "ClusterName": ClusterName,
                    "ClusterStatus": "InService",
                    "Orchestrator": {},  # No EKS = Slurm
                }
            else:
                return {
                    "ClusterName": ClusterName,
                    "ClusterStatus": "InService",
                    "Orchestrator": {"Eks": {"ClusterArn": "arn:aws:eks:..."}},
                }

        mock_sm_client.describe_cluster.side_effect = describe_side_effect

        result = list_slurm_clusters(mock_sm_client)

        # Should only return Slurm clusters
        assert len(result["ClusterSummaries"]) == 1
        assert result["ClusterSummaries"][0]["ClusterName"] == "slurm-cluster-1"

    # =========================================================================
    # Update Tests - Requirement 3.1
    # =========================================================================

    def test_update_cluster_success(
        self, mock_sm_client, sample_cluster_name, sample_cluster_response
    ):
        """Test successful cluster update."""
        mock_sm_client.describe_cluster.return_value = sample_cluster_response
        mock_sm_client.update_cluster.return_value = {
            "ClusterArn": f"arn:aws:sagemaker:us-west-2:123456789012:cluster/{sample_cluster_name}"
        }

        result = update_slurm_cluster(
            mock_sm_client,
            sample_cluster_name,
            node_recovery="None",
        )

        assert "ClusterArn" in result
        mock_sm_client.update_cluster.assert_called_once()

    def test_update_cluster_in_transitional_state(
        self, mock_sm_client, sample_cluster_name
    ):
        """Test that updates are rejected when cluster is in transitional state."""
        transitional_response = {
            "ClusterName": sample_cluster_name,
            "ClusterStatus": "Updating",
        }
        mock_sm_client.describe_cluster.return_value = transitional_response

        with pytest.raises(ClusterInTransitionalStateError) as exc_info:
            update_slurm_cluster(
                mock_sm_client,
                sample_cluster_name,
                node_recovery="None",
            )

        assert "Updating" in str(exc_info.value)
        assert "cannot be updated" in str(exc_info.value)

    def test_update_cluster_with_instance_groups(
        self, mock_sm_client, sample_cluster_name, sample_cluster_response
    ):
        """Test updating cluster with new instance group configuration."""
        mock_sm_client.describe_cluster.return_value = sample_cluster_response
        mock_sm_client.update_cluster.return_value = {
            "ClusterArn": f"arn:aws:sagemaker:us-west-2:123456789012:cluster/{sample_cluster_name}"
        }

        new_instance_groups = [
            {
                "InstanceGroupName": "workers",
                "InstanceCount": 4,
                "InstanceType": "ml.p4d.24xlarge",
            }
        ]

        result = update_slurm_cluster(
            mock_sm_client,
            sample_cluster_name,
            instance_groups=new_instance_groups,
        )

        assert "ClusterArn" in result
        call_args = mock_sm_client.update_cluster.call_args
        assert call_args.kwargs["InstanceGroups"] == new_instance_groups

    # =========================================================================
    # Delete Tests - Requirement 4.1
    # =========================================================================

    def test_delete_cluster_success(self, mock_sm_client, sample_cluster_name):
        """Test successful cluster deletion."""
        mock_sm_client.delete_cluster.return_value = {}

        # Should not raise any exception
        delete_slurm_cluster(mock_sm_client, sample_cluster_name)

        mock_sm_client.delete_cluster.assert_called_once_with(
            ClusterName=sample_cluster_name
        )

    def test_delete_cluster_not_found(self, mock_sm_client, sample_cluster_name):
        """Test that ClusterNotFoundError is raised for non-existent clusters."""
        error_response = {
            "Error": {
                "Code": "ResourceNotFoundException",
                "Message": f"Cluster {sample_cluster_name} not found",
            }
        }
        mock_sm_client.delete_cluster.side_effect = ClientError(
            error_response, "DeleteCluster"
        )

        with pytest.raises(ClusterNotFoundError) as exc_info:
            delete_slurm_cluster(mock_sm_client, sample_cluster_name)

        assert sample_cluster_name in str(exc_info.value)

    # =========================================================================
    # Full Lifecycle Test
    # =========================================================================

    def test_full_crud_lifecycle(
        self, mock_sm_client, sample_cluster_config, sample_cluster_name
    ):
        """Test complete create → describe → update → delete workflow."""
        cluster_arn = f"arn:aws:sagemaker:us-west-2:123456789012:cluster/{sample_cluster_name}"

        # Step 1: Create
        mock_sm_client.create_cluster.return_value = {"ClusterArn": cluster_arn}
        create_result = create_slurm_cluster(mock_sm_client, sample_cluster_config)
        assert create_result == cluster_arn

        # Step 2: Describe
        describe_response = {
            "ClusterArn": cluster_arn,
            "ClusterName": sample_cluster_name,
            "ClusterStatus": "InService",
            "InstanceGroups": [],
            "VpcConfig": {"Subnets": [], "SecurityGroupIds": []},
            "Tags": [],
        }
        mock_sm_client.describe_cluster.return_value = describe_response
        describe_result = describe_slurm_cluster(mock_sm_client, sample_cluster_name)
        assert describe_result["ClusterName"] == sample_cluster_name
        assert describe_result["ClusterStatus"] == "InService"

        # Step 3: Update
        mock_sm_client.update_cluster.return_value = {"ClusterArn": cluster_arn}
        update_result = update_slurm_cluster(
            mock_sm_client,
            sample_cluster_name,
            node_recovery="None",
        )
        assert "ClusterArn" in update_result

        # Step 4: Delete
        mock_sm_client.delete_cluster.return_value = {}
        delete_slurm_cluster(mock_sm_client, sample_cluster_name)
        mock_sm_client.delete_cluster.assert_called_with(
            ClusterName=sample_cluster_name
        )

    # =========================================================================
    # Orchestrator Type Detection Tests
    # =========================================================================

    def test_orchestrator_type_detection_slurm(self):
        """Test that clusters without EKS orchestrator are detected as Slurm."""
        cluster_details = {
            "ClusterName": "test-cluster",
            "Orchestrator": {},
        }
        assert get_cluster_orchestrator_type(cluster_details) == "Slurm"

    def test_orchestrator_type_detection_eks(self):
        """Test that clusters with EKS orchestrator are detected as EKS."""
        cluster_details = {
            "ClusterName": "test-cluster",
            "Orchestrator": {
                "Eks": {"ClusterArn": "arn:aws:eks:us-west-2:123456789012:cluster/test"}
            },
        }
        assert get_cluster_orchestrator_type(cluster_details) == "EKS"


class TestSlurmClusterErrorHandling:
    """Integration tests for error handling paths."""

    @pytest.fixture
    def mock_sm_client(self):
        """Create a mock SageMaker client."""
        return MagicMock()

    def test_create_cluster_access_denied(self, mock_sm_client):
        """Test handling of access denied errors during creation."""
        config = SlurmClusterConfig(
            cluster_name="test-cluster",
            instance_groups=[
                InstanceGroupConfig(
                    instance_group_name="controller",
                    instance_type="ml.m5.xlarge",
                    instance_count=1,
                    execution_role="arn:aws:iam::123456789012:role/SageMakerRole",
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

        error_response = {
            "Error": {
                "Code": "AccessDeniedException",
                "Message": "User is not authorized to perform sagemaker:CreateCluster",
            }
        }
        mock_sm_client.create_cluster.side_effect = ClientError(
            error_response, "CreateCluster"
        )

        with pytest.raises(ClientError) as exc_info:
            create_slurm_cluster(mock_sm_client, config)

        assert "AccessDeniedException" in str(exc_info.value)

    def test_update_cluster_validation_error(self, mock_sm_client):
        """Test handling of validation errors during update."""
        # First describe returns InService status
        mock_sm_client.describe_cluster.return_value = {
            "ClusterName": "test-cluster",
            "ClusterStatus": "InService",
        }

        # Update fails with validation error
        error_response = {
            "Error": {
                "Code": "ValidationException",
                "Message": "Invalid instance count: must be greater than 0",
            }
        }
        mock_sm_client.update_cluster.side_effect = ClientError(
            error_response, "UpdateCluster"
        )

        with pytest.raises(ClientError) as exc_info:
            update_slurm_cluster(
                mock_sm_client,
                "test-cluster",
                instance_groups=[{"InstanceGroupName": "workers", "InstanceCount": -1}],
            )

        assert "ValidationException" in str(exc_info.value)

    def test_describe_cluster_throttling_retry(self, mock_sm_client):
        """Test that throttling errors trigger retry logic."""
        throttle_error = {
            "Error": {
                "Code": "ThrottlingException",
                "Message": "Rate exceeded",
            }
        }
        success_response = {
            "ClusterName": "test-cluster",
            "ClusterStatus": "InService",
        }

        # First call throttles, second succeeds
        mock_sm_client.describe_cluster.side_effect = [
            ClientError(throttle_error, "DescribeCluster"),
            success_response,
        ]

        result = describe_slurm_cluster(mock_sm_client, "test-cluster")

        assert result["ClusterStatus"] == "InService"
        assert mock_sm_client.describe_cluster.call_count == 2

    def test_delete_cluster_already_deleting(self, mock_sm_client):
        """Test handling when cluster is already being deleted."""
        # Simulate cluster already in Deleting state
        error_response = {
            "Error": {
                "Code": "ConflictException",
                "Message": "Cluster is already being deleted",
            }
        }
        mock_sm_client.delete_cluster.side_effect = ClientError(
            error_response, "DeleteCluster"
        )

        with pytest.raises(ClientError) as exc_info:
            delete_slurm_cluster(mock_sm_client, "test-cluster")

        assert "ConflictException" in str(exc_info.value)

    def test_update_cluster_creating_state_rejected(self, mock_sm_client):
        """Test that updates are rejected when cluster is Creating."""
        mock_sm_client.describe_cluster.return_value = {
            "ClusterName": "test-cluster",
            "ClusterStatus": "Creating",
        }

        with pytest.raises(ClusterInTransitionalStateError) as exc_info:
            update_slurm_cluster(
                mock_sm_client,
                "test-cluster",
                node_recovery="None",
            )

        assert "Creating" in str(exc_info.value)

    def test_update_cluster_deleting_state_rejected(self, mock_sm_client):
        """Test that updates are rejected when cluster is Deleting."""
        mock_sm_client.describe_cluster.return_value = {
            "ClusterName": "test-cluster",
            "ClusterStatus": "Deleting",
        }

        with pytest.raises(ClusterInTransitionalStateError) as exc_info:
            update_slurm_cluster(
                mock_sm_client,
                "test-cluster",
                node_recovery="None",
            )

        assert "Deleting" in str(exc_info.value)

    def test_update_cluster_rolling_back_state_rejected(self, mock_sm_client):
        """Test that updates are rejected when cluster is RollingBack."""
        mock_sm_client.describe_cluster.return_value = {
            "ClusterName": "test-cluster",
            "ClusterStatus": "RollingBack",
        }

        with pytest.raises(ClusterInTransitionalStateError) as exc_info:
            update_slurm_cluster(
                mock_sm_client,
                "test-cluster",
                node_recovery="None",
            )

        assert "RollingBack" in str(exc_info.value)
