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
Utility functions for SageMaker HyperPod Slurm cluster operations.

This module provides wrapper functions for the SageMaker API operations
related to Slurm-orchestrated HyperPod clusters. It includes functions for
CRUD operations, node management, and cluster status monitoring.

All functions use cross-platform compatible timeout mechanisms (threading.Timer
or boto3 built-in timeouts) instead of Unix-specific signals.
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional

from botocore.client import BaseClient
from botocore.exceptions import ClientError

from sagemaker.hyperpod.cli.slurm_cluster_config import SlurmClusterConfig

logger = logging.getLogger(__name__)

# Cluster status constants
CLUSTER_STATUS_IN_SERVICE = "InService"
CLUSTER_STATUS_CREATING = "Creating"
CLUSTER_STATUS_UPDATING = "Updating"
CLUSTER_STATUS_DELETING = "Deleting"
CLUSTER_STATUS_FAILED = "Failed"
CLUSTER_STATUS_ROLLING_BACK = "RollingBack"
CLUSTER_STATUS_SYSTEM_UPDATING = "SystemUpdating"

# Transitional states where updates are not allowed
TRANSITIONAL_STATES = {
    CLUSTER_STATUS_CREATING,
    CLUSTER_STATUS_UPDATING,
    CLUSTER_STATUS_DELETING,
    CLUSTER_STATUS_ROLLING_BACK,
    CLUSTER_STATUS_SYSTEM_UPDATING,
}

# Terminal states for wait operations
TERMINAL_STATES = {
    CLUSTER_STATUS_IN_SERVICE,
    CLUSTER_STATUS_FAILED,
}

# Retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY_SECONDS = 1
DEFAULT_MAX_DELAY_SECONDS = 30
RETRYABLE_ERROR_CODES = {
    "ThrottlingException",
    "ServiceUnavailable",
    "InternalServerError",
}


class SlurmClusterError(Exception):
    """Base exception for Slurm cluster operations."""

    pass


class ClusterNotFoundError(SlurmClusterError):
    """Exception raised when a cluster is not found."""

    pass


class ClusterInTransitionalStateError(SlurmClusterError):
    """Exception raised when a cluster is in a transitional state."""

    pass


class ClusterOperationTimeoutError(SlurmClusterError):
    """Exception raised when a cluster operation times out."""

    pass


class InvalidNodeIdError(SlurmClusterError):
    """Exception raised when invalid node IDs are provided."""

    pass


def _calculate_backoff_delay(
    attempt: int,
    base_delay: float = DEFAULT_BASE_DELAY_SECONDS,
    max_delay: float = DEFAULT_MAX_DELAY_SECONDS,
) -> float:
    """
    Calculate exponential backoff delay for retries.

    Args:
        attempt: The current retry attempt number (0-indexed).
        base_delay: Base delay in seconds.
        max_delay: Maximum delay in seconds.

    Returns:
        The delay in seconds before the next retry.
    """
    delay = base_delay * (2**attempt)
    return min(delay, max_delay)


def _is_retryable_error(error: ClientError) -> bool:
    """
    Check if a ClientError is retryable.

    Args:
        error: The ClientError to check.

    Returns:
        True if the error is retryable, False otherwise.
    """
    error_code = error.response.get("Error", {}).get("Code", "")
    return error_code in RETRYABLE_ERROR_CODES


def _retry_with_backoff(
    func,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY_SECONDS,
    max_delay: float = DEFAULT_MAX_DELAY_SECONDS,
):
    """
    Execute a function with exponential backoff retry logic.

    Args:
        func: The function to execute.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds between retries.
        max_delay: Maximum delay in seconds between retries.

    Returns:
        The result of the function call.

    Raises:
        ClientError: If all retries are exhausted or error is not retryable.
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except ClientError as e:
            last_error = e
            if not _is_retryable_error(e) or attempt >= max_retries:
                raise

            delay = _calculate_backoff_delay(attempt, base_delay, max_delay)
            logger.debug(
                f"Retryable error encountered: {e.response['Error']['Code']}. "
                f"Retrying in {delay:.1f} seconds (attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(delay)

    # This should not be reached, but just in case
    if last_error:
        raise last_error


def create_slurm_cluster(
    sm_client: BaseClient,
    config: SlurmClusterConfig,
) -> str:
    """
    Create a Slurm-orchestrated HyperPod cluster.

    Args:
        sm_client: Boto3 SageMaker client.
        config: SlurmClusterConfig object containing cluster configuration.

    Returns:
        The ARN of the created cluster.

    Raises:
        ClientError: If the API call fails.
        SlurmClusterError: If cluster creation fails.
    """
    logger.info(f"Creating Slurm cluster: {config.cluster_name}")

    request = config.to_create_cluster_request()
    logger.debug(f"CreateCluster request: {request}")

    def _create():
        return sm_client.create_cluster(**request)

    try:
        response = _retry_with_backoff(_create)
        cluster_arn = response.get("ClusterArn", "")
        logger.info(f"Cluster creation initiated. ARN: {cluster_arn}")
        return cluster_arn
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        error_message = e.response.get("Error", {}).get("Message", "")
        logger.error(f"Failed to create cluster: {error_code} - {error_message}")
        raise


def describe_slurm_cluster(
    sm_client: BaseClient,
    cluster_name: str,
) -> Dict[str, Any]:
    """
    Describe a Slurm-orchestrated HyperPod cluster.

    Args:
        sm_client: Boto3 SageMaker client.
        cluster_name: Name of the cluster to describe.

    Returns:
        Dictionary containing cluster details.

    Raises:
        ClusterNotFoundError: If the cluster is not found.
        ClientError: If the API call fails.
    """
    logger.debug(f"Describing cluster: {cluster_name}")

    def _describe():
        return sm_client.describe_cluster(ClusterName=cluster_name)

    try:
        response = _retry_with_backoff(_describe)
        return response
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "ResourceNotFoundException":
            raise ClusterNotFoundError(f"Cluster '{cluster_name}' not found")
        raise


def list_slurm_clusters(
    sm_client: BaseClient,
    max_results: Optional[int] = None,
    next_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List Slurm-orchestrated HyperPod clusters.

    This function lists all clusters and filters to return only
    Slurm-orchestrated clusters (those without EKS orchestrator).

    Args:
        sm_client: Boto3 SageMaker client.
        max_results: Maximum number of results to return.
        next_token: Token for pagination.

    Returns:
        Dictionary containing list of Slurm clusters and pagination token.

    Raises:
        ClientError: If the API call fails.
    """
    logger.debug("Listing Slurm clusters")

    def _list():
        params = {}
        if max_results:
            params["MaxResults"] = max_results
        if next_token:
            params["NextToken"] = next_token
        return sm_client.list_clusters(**params)

    try:
        response = _retry_with_backoff(_list)

        # Filter to only include Slurm clusters
        all_clusters = response.get("ClusterSummaries", [])
        slurm_clusters = []

        for cluster_summary in all_clusters:
            cluster_name = cluster_summary.get("ClusterName", "")
            try:
                details = describe_slurm_cluster(sm_client, cluster_name)
                if get_cluster_orchestrator_type(details) == "Slurm":
                    slurm_clusters.append(cluster_summary)
            except (ClusterNotFoundError, ClientError) as e:
                logger.debug(f"Could not describe cluster {cluster_name}: {e}")
                continue

        return {
            "ClusterSummaries": slurm_clusters,
            "NextToken": response.get("NextToken"),
        }
    except ClientError:
        raise


def update_slurm_cluster(
    sm_client: BaseClient,
    cluster_name: str,
    instance_groups: Optional[List[Dict[str, Any]]] = None,
    instance_groups_to_delete: Optional[List[str]] = None,
    node_recovery: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Update a Slurm-orchestrated HyperPod cluster.

    Args:
        sm_client: Boto3 SageMaker client.
        cluster_name: Name of the cluster to update.
        instance_groups: List of instance group configurations to update.
        instance_groups_to_delete: List of instance group names to delete.
        node_recovery: Node recovery setting ("Automatic" or "None").

    Returns:
        Dictionary containing update response.

    Raises:
        ClusterNotFoundError: If the cluster is not found.
        ClusterInTransitionalStateError: If cluster is in a transitional state.
        ClientError: If the API call fails.
    """
    logger.info(f"Updating cluster: {cluster_name}")

    # First, check cluster status
    cluster_details = describe_slurm_cluster(sm_client, cluster_name)
    current_status = cluster_details.get("ClusterStatus", "")

    if current_status in TRANSITIONAL_STATES:
        raise ClusterInTransitionalStateError(
            f"Cluster '{cluster_name}' is in state '{current_status}' and cannot be updated. "
            f"Please wait for the current operation to complete."
        )

    # Build update request
    request: Dict[str, Any] = {"ClusterName": cluster_name}

    if instance_groups:
        request["InstanceGroups"] = instance_groups

    if instance_groups_to_delete:
        request["InstanceGroupsToDelete"] = instance_groups_to_delete

    if node_recovery:
        request["NodeRecovery"] = node_recovery

    logger.debug(f"UpdateCluster request: {request}")

    def _update():
        return sm_client.update_cluster(**request)

    try:
        response = _retry_with_backoff(_update)
        logger.info(f"Cluster update initiated for: {cluster_name}")
        return response
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "ResourceNotFoundException":
            raise ClusterNotFoundError(f"Cluster '{cluster_name}' not found")
        raise


def delete_slurm_cluster(
    sm_client: BaseClient,
    cluster_name: str,
) -> None:
    """
    Delete a Slurm-orchestrated HyperPod cluster.

    Args:
        sm_client: Boto3 SageMaker client.
        cluster_name: Name of the cluster to delete.

    Raises:
        ClusterNotFoundError: If the cluster is not found.
        ClientError: If the API call fails.
    """
    logger.info(f"Deleting cluster: {cluster_name}")

    def _delete():
        return sm_client.delete_cluster(ClusterName=cluster_name)

    try:
        _retry_with_backoff(_delete)
        logger.info(f"Cluster deletion initiated for: {cluster_name}")
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "ResourceNotFoundException":
            raise ClusterNotFoundError(f"Cluster '{cluster_name}' not found")
        raise


def update_cluster_software(
    sm_client: BaseClient,
    cluster_name: str,
) -> Dict[str, Any]:
    """
    Update cluster software (AMI patches) for a Slurm cluster.

    Args:
        sm_client: Boto3 SageMaker client.
        cluster_name: Name of the cluster to update.

    Returns:
        Dictionary containing update response.

    Raises:
        ClusterNotFoundError: If the cluster is not found.
        ClientError: If the API call fails.
    """
    logger.info(f"Updating cluster software for: {cluster_name}")

    def _update_software():
        return sm_client.update_cluster_software(ClusterName=cluster_name)

    try:
        response = _retry_with_backoff(_update_software)
        logger.info(f"Cluster software update initiated for: {cluster_name}")
        return response
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "ResourceNotFoundException":
            raise ClusterNotFoundError(f"Cluster '{cluster_name}' not found")
        raise


def wait_for_cluster_status(
    sm_client: BaseClient,
    cluster_name: str,
    target_statuses: List[str],
    timeout_seconds: int = 3600,
    poll_interval_seconds: int = 30,
) -> str:
    """
    Poll cluster status until it reaches a target state.

    Uses cross-platform compatible timeout mechanism (threading.Timer)
    instead of Unix-specific signals.

    Args:
        sm_client: Boto3 SageMaker client.
        cluster_name: Name of the cluster to monitor.
        target_statuses: List of target statuses to wait for.
        timeout_seconds: Maximum time to wait in seconds (default: 3600).
        poll_interval_seconds: Interval between status checks (default: 30).

    Returns:
        The final cluster status.

    Raises:
        ClusterNotFoundError: If the cluster is not found.
        ClusterOperationTimeoutError: If the timeout is exceeded.
        ClientError: If the API call fails.
    """
    logger.info(
        f"Waiting for cluster '{cluster_name}' to reach status: {target_statuses}"
    )

    timeout_event = threading.Event()
    timed_out = [False]  # Use list to allow modification in nested function

    def _timeout_handler():
        timed_out[0] = True
        timeout_event.set()

    # Start timeout timer
    timer = threading.Timer(timeout_seconds, _timeout_handler)
    timer.daemon = True
    timer.start()

    try:
        start_time = time.time()

        while not timed_out[0]:
            try:
                cluster_details = describe_slurm_cluster(sm_client, cluster_name)
                current_status = cluster_details.get("ClusterStatus", "")

                elapsed = time.time() - start_time
                logger.debug(
                    f"Cluster status: {current_status} (elapsed: {elapsed:.0f}s)"
                )

                if current_status in target_statuses:
                    logger.info(
                        f"Cluster '{cluster_name}' reached status: {current_status}"
                    )
                    return current_status

                # Check for failure state
                if current_status == CLUSTER_STATUS_FAILED:
                    failure_message = cluster_details.get("FailureMessage", "Unknown")
                    logger.error(
                        f"Cluster '{cluster_name}' failed: {failure_message}"
                    )
                    return current_status

            except ClusterNotFoundError:
                # Cluster might have been deleted
                if CLUSTER_STATUS_DELETING in target_statuses:
                    logger.info(f"Cluster '{cluster_name}' has been deleted")
                    return "Deleted"
                raise

            # Wait before next poll
            timeout_event.wait(poll_interval_seconds)

        # Timeout occurred
        raise ClusterOperationTimeoutError(
            f"Timeout waiting for cluster '{cluster_name}' to reach status "
            f"{target_statuses} after {timeout_seconds} seconds"
        )

    finally:
        timer.cancel()


def list_cluster_nodes(
    sm_client: BaseClient,
    cluster_name: str,
    max_results: Optional[int] = None,
    next_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List all nodes in a Slurm cluster.

    Args:
        sm_client: Boto3 SageMaker client.
        cluster_name: Name of the cluster.
        max_results: Maximum number of results to return.
        next_token: Token for pagination.

    Returns:
        Dictionary containing list of cluster nodes and pagination token.

    Raises:
        ClusterNotFoundError: If the cluster is not found.
        ClientError: If the API call fails.
    """
    logger.debug(f"Listing nodes for cluster: {cluster_name}")

    def _list_nodes():
        params: Dict[str, Any] = {"ClusterName": cluster_name}
        if max_results:
            params["MaxResults"] = max_results
        if next_token:
            params["NextToken"] = next_token
        return sm_client.list_cluster_nodes(**params)

    try:
        response = _retry_with_backoff(_list_nodes)
        return response
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "ResourceNotFoundException":
            raise ClusterNotFoundError(f"Cluster '{cluster_name}' not found")
        raise


def describe_cluster_node(
    sm_client: BaseClient,
    cluster_name: str,
    node_id: str,
) -> Dict[str, Any]:
    """
    Describe a specific node in a Slurm cluster.

    Args:
        sm_client: Boto3 SageMaker client.
        cluster_name: Name of the cluster.
        node_id: ID of the node to describe.

    Returns:
        Dictionary containing node details.

    Raises:
        ClusterNotFoundError: If the cluster is not found.
        InvalidNodeIdError: If the node is not found.
        ClientError: If the API call fails.
    """
    logger.debug(f"Describing node {node_id} in cluster: {cluster_name}")

    def _describe_node():
        return sm_client.describe_cluster_node(
            ClusterName=cluster_name,
            NodeId=node_id,
        )

    try:
        response = _retry_with_backoff(_describe_node)
        return response
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        error_message = e.response.get("Error", {}).get("Message", "")

        if error_code == "ResourceNotFoundException":
            if "cluster" in error_message.lower():
                raise ClusterNotFoundError(f"Cluster '{cluster_name}' not found")
            else:
                raise InvalidNodeIdError(
                    f"Node '{node_id}' not found in cluster '{cluster_name}'"
                )
        raise


def batch_delete_cluster_nodes(
    sm_client: BaseClient,
    cluster_name: str,
    node_ids: List[str],
) -> Dict[str, Any]:
    """
    Remove specific nodes from a Slurm cluster.

    Args:
        sm_client: Boto3 SageMaker client.
        cluster_name: Name of the cluster.
        node_ids: List of node IDs to remove.

    Returns:
        Dictionary containing deletion response with failed and successful nodes.

    Raises:
        ClusterNotFoundError: If the cluster is not found.
        InvalidNodeIdError: If any node IDs are invalid.
        ClientError: If the API call fails.
    """
    logger.info(f"Deleting nodes {node_ids} from cluster: {cluster_name}")

    def _batch_delete():
        return sm_client.batch_delete_cluster_nodes(
            ClusterName=cluster_name,
            NodeIds=node_ids,
        )

    try:
        response = _retry_with_backoff(_batch_delete)

        # Check for failed deletions
        failed = response.get("Failed", [])
        if failed:
            failed_ids = [f.get("NodeId", "unknown") for f in failed]
            logger.warning(f"Some nodes failed to delete: {failed_ids}")

        successful = response.get("Successful", [])
        if successful:
            logger.info(f"Successfully deleted {len(successful)} nodes")

        return response
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "ResourceNotFoundException":
            raise ClusterNotFoundError(f"Cluster '{cluster_name}' not found")
        raise


def get_cluster_orchestrator_type(
    cluster_details: Dict[str, Any],
) -> str:
    """
    Determine if a cluster is Slurm or EKS orchestrated.

    Args:
        cluster_details: Dictionary containing cluster details from DescribeCluster.

    Returns:
        "Slurm" if the cluster is Slurm-orchestrated, "EKS" if EKS-orchestrated.
    """
    orchestrator = cluster_details.get("Orchestrator", {})

    # If there's an EKS configuration, it's an EKS cluster
    if orchestrator.get("Eks"):
        return "EKS"

    # Otherwise, it's a Slurm cluster
    return "Slurm"


def is_cluster_in_transitional_state(status: str) -> bool:
    """
    Check if a cluster status is a transitional state.

    Args:
        status: The cluster status to check.

    Returns:
        True if the status is a transitional state, False otherwise.
    """
    return status in TRANSITIONAL_STATES


def get_cluster_status(
    sm_client: BaseClient,
    cluster_name: str,
) -> str:
    """
    Get the current status of a cluster.

    Args:
        sm_client: Boto3 SageMaker client.
        cluster_name: Name of the cluster.

    Returns:
        The current cluster status.

    Raises:
        ClusterNotFoundError: If the cluster is not found.
        ClientError: If the API call fails.
    """
    cluster_details = describe_slurm_cluster(sm_client, cluster_name)
    return cluster_details.get("ClusterStatus", "")
