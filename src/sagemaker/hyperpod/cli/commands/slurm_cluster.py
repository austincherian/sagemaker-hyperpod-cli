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
CLI commands for SageMaker HyperPod Slurm cluster operations.

This module provides Click commands for managing Slurm-orchestrated
HyperPod clusters, including create, describe, list, update, delete,
software update, node management, and scale-down operations.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import boto3
import click
import yaml
from botocore.exceptions import ClientError, NoCredentialsError
from tabulate import tabulate

from sagemaker.hyperpod.cli.slurm_cluster_config import SlurmClusterConfig
from sagemaker.hyperpod.cli.slurm_cluster_utils import (
    ClusterInTransitionalStateError,
    ClusterNotFoundError,
    ClusterOperationTimeoutError,
    InvalidNodeIdError,
    batch_delete_cluster_nodes,
    create_slurm_cluster,
    delete_slurm_cluster,
    describe_cluster_node,
    describe_slurm_cluster,
    get_cluster_orchestrator_type,
    list_cluster_nodes,
    list_slurm_clusters,
    update_cluster_software,
    update_slurm_cluster,
    wait_for_cluster_status,
    CLUSTER_STATUS_IN_SERVICE,
    CLUSTER_STATUS_FAILED,
)
from sagemaker.hyperpod.cli.validators.slurm_validator import SlurmClusterValidator
from sagemaker.hyperpod.cli.utils import setup_logger


logger = logging.getLogger(__name__)


# Error messages for user-friendly output
ERROR_MESSAGES = {
    "NoCredentialsError": (
        "AWS credentials not found or invalid. "
        "Run 'aws configure' or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
    ),
    "AccessDeniedException": (
        "Access denied. Ensure your IAM role has the following permissions: "
        "sagemaker:CreateCluster, sagemaker:DescribeCluster, sagemaker:UpdateCluster, "
        "sagemaker:DeleteCluster, sagemaker:ListClusters, sagemaker:ListClusterNodes, "
        "sagemaker:DescribeClusterNode, sagemaker:BatchDeleteClusterNodes, "
        "sagemaker:UpdateClusterSoftware"
    ),
    "ResourceNotFoundException": "Cluster '{cluster_name}' not found in region '{region}'.",
    "ValidationException": "Invalid configuration: {message}",
    "ThrottlingException": "API rate limit exceeded. Please try again later.",
}


def _get_sagemaker_client(region: Optional[str] = None):
    """
    Create a SageMaker client with optional region override.

    Args:
        region: AWS region. If None, uses default from environment/config.

    Returns:
        boto3 SageMaker client.

    Raises:
        NoCredentialsError: If AWS credentials are not configured.
    """
    try:
        if region:
            return boto3.client("sagemaker", region_name=region)
        return boto3.client("sagemaker")
    except NoCredentialsError:
        raise click.ClickException(ERROR_MESSAGES["NoCredentialsError"])


def _load_config_from_file(config_path: Optional[str] = None) -> dict:
    """
    Load cluster configuration from a YAML file.

    Args:
        config_path: Path to config file. If None, looks for config.yaml in current directory.

    Returns:
        Dictionary containing the configuration.

    Raises:
        click.ClickException: If config file is not found or invalid.
    """
    if config_path is None:
        config_path = os.path.join(os.getcwd(), "config.yaml")

    config_file = Path(config_path)
    if not config_file.exists():
        raise click.ClickException(
            f"Configuration file not found: {config_path}. "
            f"Run 'hyp init slurm-cluster' to create a configuration file."
        )

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise click.ClickException(f"Invalid YAML in configuration file: {e}")


def _handle_client_error(e: ClientError, cluster_name: str = "", region: str = ""):
    """
    Handle boto3 ClientError and raise appropriate click exception.

    Args:
        e: The ClientError to handle.
        cluster_name: Cluster name for error messages.
        region: Region for error messages.

    Raises:
        click.ClickException: With user-friendly error message.
    """
    error_code = e.response.get("Error", {}).get("Code", "")
    error_message = e.response.get("Error", {}).get("Message", "")

    if error_code == "AccessDeniedException":
        raise click.ClickException(ERROR_MESSAGES["AccessDeniedException"])
    elif error_code == "ResourceNotFoundException":
        raise click.ClickException(
            ERROR_MESSAGES["ResourceNotFoundException"].format(
                cluster_name=cluster_name, region=region or "default"
            )
        )
    elif error_code == "ValidationException":
        raise click.ClickException(
            ERROR_MESSAGES["ValidationException"].format(message=error_message)
        )
    elif error_code == "ThrottlingException":
        raise click.ClickException(ERROR_MESSAGES["ThrottlingException"])
    else:
        raise click.ClickException(f"AWS API error: {error_code} - {error_message}")


def _format_cluster_details_table(cluster_details: dict) -> str:
    """
    Format cluster details as a table for display.

    Args:
        cluster_details: Dictionary containing cluster details.

    Returns:
        Formatted table string.
    """
    table_data = []
    for key, value in cluster_details.items():
        if isinstance(value, (dict, list)):
            formatted_value = json.dumps(value, indent=2, default=str)
        else:
            formatted_value = str(value)
        table_data.append([key, formatted_value])

    return tabulate(table_data, headers=["Field", "Value"], tablefmt="presto")


def _format_cluster_details_json(cluster_details: dict) -> str:
    """
    Format cluster details as JSON for display.

    Args:
        cluster_details: Dictionary containing cluster details.

    Returns:
        JSON string.
    """
    return json.dumps(cluster_details, indent=2, default=str)


# =============================================================================
# Create Command - Requirement 1.1
# =============================================================================


@click.command("slurm-cluster")
@click.option(
    "--cluster-name",
    type=str,
    help="Cluster name (overrides value in config.yaml)",
)
@click.option(
    "--region",
    type=str,
    help="AWS region",
)
@click.option(
    "--wait",
    is_flag=True,
    help="Wait for cluster to reach InService or Failed state",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging",
)
def slurm_cluster_create(
    cluster_name: Optional[str],
    region: Optional[str],
    wait: bool,
    debug: bool,
):
    """
    Create a new Slurm-orchestrated HyperPod cluster.

    Creates a SageMaker HyperPod cluster using the Slurm workload manager.
    Configuration is loaded from config.yaml in the current directory.

    .. dropdown:: Usage Examples
       :open:

       .. code-block:: bash

          # Create cluster using config.yaml in current directory
          hyp create slurm-cluster

          # Create with custom cluster name
          hyp create slurm-cluster --cluster-name my-cluster

          # Create and wait for completion
          hyp create slurm-cluster --wait

          # Create in specific region
          hyp create slurm-cluster --region us-west-2
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)

    # Load configuration
    try:
        config_dict = _load_config_from_file()
    except click.ClickException:
        raise

    # Override cluster name if provided
    if cluster_name:
        config_dict["cluster_name"] = cluster_name

    # Override region if provided
    if region:
        config_dict["region"] = region

    # Validate configuration
    validator = SlurmClusterValidator()
    validation_result = validator.validate_config(config_dict=config_dict)

    if not validation_result.is_valid:
        error_summary = validator.get_validation_errors_summary(validation_result)
        raise click.ClickException(error_summary)

    # Parse configuration
    try:
        config = SlurmClusterConfig(**config_dict)
    except Exception as e:
        raise click.ClickException(f"Invalid configuration: {e}")

    # Create SageMaker client
    sm_client = _get_sagemaker_client(region or config.region)

    # Create cluster
    try:
        click.echo(f"Creating Slurm cluster: {config.cluster_name}")
        cluster_arn = create_slurm_cluster(sm_client, config)
        click.secho(f"✓ Cluster creation initiated", fg="green")
        click.echo(f"Cluster ARN: {cluster_arn}")

        if wait:
            click.echo("Waiting for cluster to reach InService state...")
            final_status = wait_for_cluster_status(
                sm_client,
                config.cluster_name,
                [CLUSTER_STATUS_IN_SERVICE, CLUSTER_STATUS_FAILED],
            )

            if final_status == CLUSTER_STATUS_IN_SERVICE:
                click.secho(f"✓ Cluster is now InService", fg="green")
            elif final_status == CLUSTER_STATUS_FAILED:
                click.secho(f"✗ Cluster creation failed", fg="red")
                # Get failure details
                details = describe_slurm_cluster(sm_client, config.cluster_name)
                failure_msg = details.get("FailureMessage", "Unknown error")
                click.echo(f"Failure reason: {failure_msg}")
                raise click.ClickException("Cluster creation failed")

    except ClusterOperationTimeoutError as e:
        raise click.ClickException(str(e))
    except ClientError as e:
        _handle_client_error(e, config.cluster_name, region)


# =============================================================================
# Describe Command - Requirement 2.1
# =============================================================================


@click.command("slurm-cluster")
@click.argument("cluster_name", required=True)
@click.option(
    "--output",
    type=click.Choice(["json", "table"]),
    default="table",
    help="Output format (json or table)",
)
@click.option(
    "--region",
    type=str,
    help="AWS region",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging",
)
def slurm_cluster_describe(
    cluster_name: str,
    output: str,
    region: Optional[str],
    debug: bool,
):
    """
    Describe a Slurm-orchestrated HyperPod cluster.

    Shows detailed information about a cluster including status,
    instance groups, VPC configuration, and tags.

    .. dropdown:: Usage Examples
       :open:

       .. code-block:: bash

          # Describe cluster in table format
          hyp describe slurm-cluster my-cluster

          # Describe cluster in JSON format
          hyp describe slurm-cluster my-cluster --output json

          # Describe cluster in specific region
          hyp describe slurm-cluster my-cluster --region us-west-2
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)

    sm_client = _get_sagemaker_client(region)

    try:
        cluster_details = describe_slurm_cluster(sm_client, cluster_name)

        # Verify it's a Slurm cluster
        orchestrator_type = get_cluster_orchestrator_type(cluster_details)
        if orchestrator_type != "Slurm":
            raise click.ClickException(
                f"Cluster '{cluster_name}' is an EKS cluster, not a Slurm cluster. "
                f"Use 'hyp describe cluster-stack' for EKS clusters."
            )

        click.echo(f"📋 Cluster Details for: {cluster_name}")
        click.echo(f"Status: {cluster_details.get('ClusterStatus', 'Unknown')}")
        click.echo()

        if output == "json":
            click.echo(_format_cluster_details_json(cluster_details))
        else:
            click.echo(_format_cluster_details_table(cluster_details))

    except ClusterNotFoundError:
        raise click.ClickException(
            f"Cluster '{cluster_name}' not found in region '{region or 'default'}'"
        )
    except ClientError as e:
        _handle_client_error(e, cluster_name, region)


# =============================================================================
# List Command - Requirement 10.1
# =============================================================================


@click.command("slurm-cluster")
@click.option(
    "--region",
    type=str,
    help="AWS region",
)
@click.option(
    "--output",
    type=click.Choice(["json", "table"]),
    default="table",
    help="Output format (json or table)",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging",
)
def slurm_cluster_list(
    region: Optional[str],
    output: str,
    debug: bool,
):
    """
    List Slurm-orchestrated HyperPod clusters.

    Lists all Slurm clusters in the specified region. EKS clusters
    are filtered out and not displayed.

    .. dropdown:: Usage Examples
       :open:

       .. code-block:: bash

          # List all Slurm clusters
          hyp list slurm-cluster

          # List clusters in specific region
          hyp list slurm-cluster --region us-west-2

          # List clusters in JSON format
          hyp list slurm-cluster --output json
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)

    sm_client = _get_sagemaker_client(region)

    try:
        result = list_slurm_clusters(sm_client)
        clusters = result.get("ClusterSummaries", [])

        if not clusters:
            click.echo("No Slurm clusters found.")
            return

        click.echo(f"📋 Slurm Clusters ({len(clusters)} found)")
        click.echo()

        if output == "json":
            click.echo(json.dumps(clusters, indent=2, default=str))
        else:
            table_data = []
            for cluster in clusters:
                table_data.append([
                    cluster.get("ClusterName", ""),
                    cluster.get("ClusterStatus", ""),
                    str(cluster.get("CreationTime", "")),
                ])
            click.echo(tabulate(
                table_data,
                headers=["Cluster Name", "Status", "Creation Time"],
                tablefmt="presto"
            ))

    except ClientError as e:
        _handle_client_error(e, region=region)


# =============================================================================
# Update Command - Requirement 3.1
# =============================================================================


@click.command("slurm-cluster")
@click.option(
    "--cluster-name",
    required=True,
    help="Name of the cluster to update",
)
@click.option(
    "--instance-groups",
    type=str,
    help="Instance groups JSON string",
)
@click.option(
    "--instance-groups-to-delete",
    type=str,
    help="Instance group names to delete (JSON array)",
)
@click.option(
    "--node-recovery",
    type=click.Choice(["Automatic", "None"]),
    help="Node recovery setting",
)
@click.option(
    "--region",
    type=str,
    help="AWS region",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging",
)
def slurm_cluster_update(
    cluster_name: str,
    instance_groups: Optional[str],
    instance_groups_to_delete: Optional[str],
    node_recovery: Optional[str],
    region: Optional[str],
    debug: bool,
):
    """
    Update a Slurm-orchestrated HyperPod cluster.

    Modifies cluster settings such as instance groups and node recovery.
    At least one update parameter must be provided.

    .. dropdown:: Usage Examples
       :open:

       .. code-block:: bash

          # Update node recovery setting
          hyp update slurm-cluster --cluster-name my-cluster --node-recovery Automatic

          # Update instance groups
          hyp update slurm-cluster --cluster-name my-cluster \\
              --instance-groups '[{"InstanceGroupName": "workers", "InstanceCount": 4}]'

          # Delete instance groups
          hyp update slurm-cluster --cluster-name my-cluster \\
              --instance-groups-to-delete '["old-workers"]'
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)

    # Validate that at least one update parameter is provided
    if not any([instance_groups, instance_groups_to_delete, node_recovery]):
        raise click.ClickException(
            "At least one of --instance-groups, --instance-groups-to-delete, "
            "or --node-recovery must be provided"
        )

    sm_client = _get_sagemaker_client(region)

    # Parse JSON parameters
    parsed_instance_groups = None
    parsed_instance_groups_to_delete = None

    if instance_groups:
        try:
            parsed_instance_groups = json.loads(instance_groups)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid JSON for --instance-groups: {e}")

    if instance_groups_to_delete:
        try:
            parsed_instance_groups_to_delete = json.loads(instance_groups_to_delete)
        except json.JSONDecodeError as e:
            raise click.ClickException(
                f"Invalid JSON for --instance-groups-to-delete: {e}"
            )

    try:
        click.echo(f"Updating cluster: {cluster_name}")
        result = update_slurm_cluster(
            sm_client,
            cluster_name,
            instance_groups=parsed_instance_groups,
            instance_groups_to_delete=parsed_instance_groups_to_delete,
            node_recovery=node_recovery,
        )
        click.secho(f"✓ Cluster update initiated", fg="green")
        click.echo(f"Cluster ARN: {result.get('ClusterArn', '')}")

    except ClusterNotFoundError:
        raise click.ClickException(
            f"Cluster '{cluster_name}' not found in region '{region or 'default'}'"
        )
    except ClusterInTransitionalStateError as e:
        raise click.ClickException(str(e))
    except ClientError as e:
        _handle_client_error(e, cluster_name, region)


# =============================================================================
# Delete Command - Requirement 4.1
# =============================================================================


@click.command("slurm-cluster")
@click.argument("cluster_name", required=True)
@click.option(
    "--region",
    required=True,
    help="AWS region",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging",
)
def slurm_cluster_delete(
    cluster_name: str,
    region: str,
    debug: bool,
):
    """
    Delete a Slurm-orchestrated HyperPod cluster.

    Removes the specified cluster and all associated resources.
    This operation cannot be undone.

    .. dropdown:: Usage Examples
       :open:

       .. code-block:: bash

          # Delete a cluster
          hyp delete slurm-cluster my-cluster --region us-west-2
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)

    sm_client = _get_sagemaker_client(region)

    try:
        # First check if cluster exists and is a Slurm cluster
        cluster_details = describe_slurm_cluster(sm_client, cluster_name)
        orchestrator_type = get_cluster_orchestrator_type(cluster_details)

        if orchestrator_type != "Slurm":
            raise click.ClickException(
                f"Cluster '{cluster_name}' is an EKS cluster, not a Slurm cluster. "
                f"Use 'hyp delete cluster-stack' for EKS clusters."
            )

        current_status = cluster_details.get("ClusterStatus", "")
        if current_status == "Deleting":
            click.echo(f"Cluster '{cluster_name}' is already being deleted.")
            return

        click.echo(f"Deleting cluster: {cluster_name}")
        delete_slurm_cluster(sm_client, cluster_name)
        click.secho(f"✓ Cluster deletion initiated", fg="green")

    except ClusterNotFoundError:
        raise click.ClickException(
            f"Cluster '{cluster_name}' not found in region '{region}'"
        )
    except ClientError as e:
        _handle_client_error(e, cluster_name, region)


# =============================================================================
# Update Software Command - Requirement 5.1
# =============================================================================


@click.command("slurm-cluster")
@click.argument("cluster_name", required=True)
@click.option(
    "--region",
    type=str,
    help="AWS region",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging",
)
def slurm_cluster_update_software(
    cluster_name: str,
    region: Optional[str],
    debug: bool,
):
    """
    Update cluster software (AMI patches) for a Slurm cluster.

    Applies the latest AMI patches to cluster nodes.

    .. dropdown:: Usage Examples
       :open:

       .. code-block:: bash

          # Update cluster software
          hyp update-software slurm-cluster my-cluster

          # Update in specific region
          hyp update-software slurm-cluster my-cluster --region us-west-2
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)

    sm_client = _get_sagemaker_client(region)

    try:
        click.echo(f"Updating software for cluster: {cluster_name}")
        result = update_cluster_software(sm_client, cluster_name)
        click.secho(f"✓ Software update initiated", fg="green")
        click.echo(f"Cluster ARN: {result.get('ClusterArn', '')}")

    except ClusterNotFoundError:
        raise click.ClickException(
            f"Cluster '{cluster_name}' not found in region '{region or 'default'}'"
        )
    except ClientError as e:
        _handle_client_error(e, cluster_name, region)


# =============================================================================
# List Nodes Command - Requirement 6.1
# =============================================================================


@click.command("slurm-cluster")
@click.argument("cluster_name", required=True)
@click.option(
    "--output",
    type=click.Choice(["json", "table"]),
    default="table",
    help="Output format (json or table)",
)
@click.option(
    "--region",
    type=str,
    help="AWS region",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging",
)
def slurm_cluster_list_nodes(
    cluster_name: str,
    output: str,
    region: Optional[str],
    debug: bool,
):
    """
    List nodes in a Slurm-orchestrated HyperPod cluster.

    Shows node ID, instance type, status, and health for each node.

    .. dropdown:: Usage Examples
       :open:

       .. code-block:: bash

          # List nodes in table format
          hyp list-nodes slurm-cluster my-cluster

          # List nodes in JSON format
          hyp list-nodes slurm-cluster my-cluster --output json
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)

    sm_client = _get_sagemaker_client(region)

    try:
        result = list_cluster_nodes(sm_client, cluster_name)
        nodes = result.get("ClusterNodeSummaries", [])

        if not nodes:
            click.echo(f"No nodes found in cluster '{cluster_name}'.")
            return

        click.echo(f"📋 Nodes in cluster '{cluster_name}' ({len(nodes)} found)")
        click.echo()

        if output == "json":
            click.echo(json.dumps(nodes, indent=2, default=str))
        else:
            table_data = []
            for node in nodes:
                instance_status = node.get("InstanceStatus", {})
                table_data.append([
                    node.get("InstanceId", ""),
                    node.get("InstanceType", ""),
                    instance_status.get("Status", ""),
                    instance_status.get("Message", ""),
                ])
            click.echo(tabulate(
                table_data,
                headers=["Node ID", "Instance Type", "Status", "Health"],
                tablefmt="presto"
            ))

    except ClusterNotFoundError:
        raise click.ClickException(
            f"Cluster '{cluster_name}' not found in region '{region or 'default'}'"
        )
    except ClientError as e:
        _handle_client_error(e, cluster_name, region)


# =============================================================================
# Describe Node Command - Requirement 6.2
# =============================================================================


@click.command("slurm-cluster")
@click.argument("cluster_name", required=True)
@click.option(
    "--node-id",
    required=True,
    help="Node ID to describe",
)
@click.option(
    "--output",
    type=click.Choice(["json", "table"]),
    default="table",
    help="Output format (json or table)",
)
@click.option(
    "--region",
    type=str,
    help="AWS region",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging",
)
def slurm_cluster_describe_node(
    cluster_name: str,
    node_id: str,
    output: str,
    region: Optional[str],
    debug: bool,
):
    """
    Describe a specific node in a Slurm cluster.

    Shows detailed information about a cluster node.

    .. dropdown:: Usage Examples
       :open:

       .. code-block:: bash

          # Describe a node
          hyp describe-node slurm-cluster my-cluster --node-id i-1234567890abcdef0

          # Describe in JSON format
          hyp describe-node slurm-cluster my-cluster --node-id i-1234567890abcdef0 --output json
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)

    sm_client = _get_sagemaker_client(region)

    try:
        node_details = describe_cluster_node(sm_client, cluster_name, node_id)

        click.echo(f"📋 Node Details for: {node_id}")
        click.echo()

        if output == "json":
            click.echo(json.dumps(node_details, indent=2, default=str))
        else:
            node_info = node_details.get("NodeDetails", {})
            click.echo(_format_cluster_details_table(node_info))

    except ClusterNotFoundError:
        raise click.ClickException(
            f"Cluster '{cluster_name}' not found in region '{region or 'default'}'"
        )
    except InvalidNodeIdError:
        raise click.ClickException(
            f"Node '{node_id}' not found in cluster '{cluster_name}'"
        )
    except ClientError as e:
        _handle_client_error(e, cluster_name, region)


# =============================================================================
# Scale Down Command - Requirement 7.1
# =============================================================================


@click.command("slurm-cluster")
@click.argument("cluster_name", required=True)
@click.option(
    "--node-ids",
    required=True,
    help="Comma-separated list of node IDs to remove",
)
@click.option(
    "--region",
    type=str,
    help="AWS region",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging",
)
def slurm_cluster_scale_down(
    cluster_name: str,
    node_ids: str,
    region: Optional[str],
    debug: bool,
):
    """
    Scale down a Slurm cluster by removing specific nodes.

    Removes the specified nodes from the cluster.

    .. dropdown:: Usage Examples
       :open:

       .. code-block:: bash

          # Remove specific nodes
          hyp scale-down slurm-cluster my-cluster --node-ids i-123,i-456

          # Remove nodes in specific region
          hyp scale-down slurm-cluster my-cluster --node-ids i-123 --region us-west-2
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)

    sm_client = _get_sagemaker_client(region)

    # Parse node IDs
    node_id_list = [nid.strip() for nid in node_ids.split(",") if nid.strip()]

    if not node_id_list:
        raise click.ClickException("No valid node IDs provided")

    try:
        click.echo(f"Removing {len(node_id_list)} node(s) from cluster: {cluster_name}")
        result = batch_delete_cluster_nodes(sm_client, cluster_name, node_id_list)

        successful = result.get("Successful", [])
        failed = result.get("Failed", [])

        if successful:
            click.secho(
                f"✓ Successfully removed {len(successful)} node(s)", fg="green"
            )

        if failed:
            click.secho(f"✗ Failed to remove {len(failed)} node(s):", fg="red")
            for failure in failed:
                click.echo(
                    f"  - {failure.get('NodeId', 'unknown')}: "
                    f"{failure.get('Message', 'Unknown error')}"
                )

    except ClusterNotFoundError:
        raise click.ClickException(
            f"Cluster '{cluster_name}' not found in region '{region or 'default'}'"
        )
    except InvalidNodeIdError as e:
        raise click.ClickException(str(e))
    except ClientError as e:
        _handle_client_error(e, cluster_name, region)
