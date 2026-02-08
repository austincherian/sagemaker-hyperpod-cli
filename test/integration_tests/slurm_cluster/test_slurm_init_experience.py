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
Integration tests for Slurm cluster init experience.

Tests the `hyp init slurm-cluster` workflow and config validation.

Requirements tested:
- 8.1: Initialize Slurm cluster configuration files via CLI
- 8.2: Create provisioning_parameters.json template
- 8.3: Prompt for reset on re-init with same template
- 8.4: Display success message with next steps
- 8.5: Support directory argument for init
- 9.1: Validate config.yaml schema
- 9.4: Display success message when validation passes
- 9.5: Display specific error messages for validation failures
"""

import os
import yaml
from contextlib import contextmanager
from pathlib import Path

import pytest
from click.testing import CliRunner

from sagemaker.hyperpod.cli.commands.init import init, validate, reset
from test.integration_tests.init.utils import (
    assert_command_succeeded,
    assert_command_failed_with_helpful_error,
    assert_config_values,
    assert_warning_displayed,
    assert_yes_no_prompt_displayed,
    assert_success_message_displayed,
)


@contextmanager
def change_directory(path):
    """Context manager for safely changing directories in tests."""
    old_cwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_cwd)


@pytest.fixture
def runner():
    """CLI test runner for invoking commands."""
    return CliRunner()


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files."""
    return str(tmp_path)


class TestSlurmClusterInit:
    """Test `hyp init slurm-cluster` command."""

    def test_init_creates_config_yaml(self, temp_dir, runner):
        """Test that init creates config.yaml with required fields (Requirement 8.1)."""
        result = runner.invoke(
            init, ["slurm-cluster", temp_dir], catch_exceptions=False
        )
        assert_command_succeeded(result)

        # Verify config.yaml was created
        config_path = Path(temp_dir) / "config.yaml"
        assert config_path.exists(), "config.yaml should be created"

        # Load and verify required fields
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Verify template and version fields
        assert config.get("template") == "slurm-cluster", "Template should be slurm-cluster"
        # Version may be stored as float (1.0) or string ("1.0") depending on YAML parsing
        version = config.get("version")
        assert version == "1.0" or version == 1.0, f"Version should be 1.0, got {version}"

        # Verify cluster_name field exists (may be None as placeholder)
        assert "cluster_name" in config, "cluster_name field should exist"

        # Verify instance_groups field exists (may be None as placeholder for user to fill)
        assert "instance_groups" in config, "instance_groups field should exist"

        # Verify vpc_config field exists (may be None as placeholder for user to fill)
        assert "vpc_config" in config, "vpc_config field should exist"

        # Verify node_recovery field defaults to Automatic
        assert config.get("node_recovery") == "Automatic", "node_recovery should default to Automatic"

        # Verify slurm_config field exists (may be None as placeholder for user to fill)
        assert "slurm_config" in config, "slurm_config field should exist"

    def test_init_creates_readme(self, temp_dir, runner):
        """Test that init creates README.md with usage instructions (Requirement 8.4)."""
        result = runner.invoke(
            init, ["slurm-cluster", temp_dir], catch_exceptions=False
        )
        assert_command_succeeded(result)

        # Verify README.md was created
        readme_path = Path(temp_dir) / "README.md"
        assert readme_path.exists(), "README.md should be created"

    def test_init_displays_success_message(self, temp_dir, runner):
        """Test that init displays success message with next steps (Requirement 8.4)."""
        result = runner.invoke(
            init, ["slurm-cluster", temp_dir], catch_exceptions=False
        )
        assert_command_succeeded(result)

        # Verify success message is displayed
        assert_success_message_displayed(result, ["✔️", "slurm-cluster", "initialized"])

        # Verify next steps are mentioned
        assert "readme" in result.output.lower(), "Should mention README for usage"

    def test_init_in_specified_directory(self, temp_dir, runner):
        """Test that init creates files in specified directory (Requirement 8.5)."""
        subdir = Path(temp_dir) / "my-cluster"

        result = runner.invoke(
            init, ["slurm-cluster", str(subdir)], catch_exceptions=False
        )
        assert_command_succeeded(result)

        # Verify files were created in the specified directory
        assert (subdir / "config.yaml").exists(), "config.yaml should be in specified directory"
        assert (subdir / "README.md").exists(), "README.md should be in specified directory"

    def test_init_creates_directory_if_not_exists(self, temp_dir, runner):
        """Test that init creates the target directory if it doesn't exist."""
        new_dir = Path(temp_dir) / "new-cluster-dir"
        assert not new_dir.exists(), "Directory should not exist initially"

        result = runner.invoke(
            init, ["slurm-cluster", str(new_dir)], catch_exceptions=False
        )
        assert_command_succeeded(result)

        assert new_dir.exists(), "Directory should be created"
        assert (new_dir / "config.yaml").exists(), "config.yaml should be created"


class TestSlurmClusterInitEdgeCases:
    """Test edge cases for slurm-cluster init command."""

    def test_double_init_same_template_prompts_override(self, temp_dir, runner):
        """Test that re-running init with same template prompts for override (Requirement 8.3)."""
        # First init
        result1 = runner.invoke(
            init, ["slurm-cluster", temp_dir], catch_exceptions=False
        )
        assert_command_succeeded(result1)

        # Second init with same template - should prompt for override
        result2 = runner.invoke(
            init, ["slurm-cluster", temp_dir],
            input="n\n",  # Answer 'no' to override prompt
            catch_exceptions=False
        )

        # Verify warning and prompt are displayed
        assert_warning_displayed(result2, ["already initialized", "override"])
        assert_yes_no_prompt_displayed(result2)
        assert "aborting init" in result2.output.lower()

    def test_double_init_same_template_accepts_override(self, temp_dir, runner):
        """Test that accepting override re-initializes successfully."""
        # First init
        result1 = runner.invoke(
            init, ["slurm-cluster", temp_dir], catch_exceptions=False
        )
        assert_command_succeeded(result1)

        # Modify config to verify it gets overwritten
        config_path = Path(temp_dir) / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        config["cluster_name"] = "modified-cluster-name"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Second init with same template - accept override
        result2 = runner.invoke(
            init, ["slurm-cluster", temp_dir],
            input="y\n",  # Answer 'yes' to override prompt
            catch_exceptions=False
        )

        assert_command_succeeded(result2)

        # Verify config was reset to defaults
        with open(config_path, "r") as f:
            new_config = yaml.safe_load(f)
        assert new_config["cluster_name"] != "modified-cluster-name", "Config should be reset"

    def test_double_init_different_template_warns_user(self, temp_dir, runner):
        """Test that re-running init with different template shows strong warning."""
        # First init with slurm-cluster
        result1 = runner.invoke(
            init, ["slurm-cluster", temp_dir], catch_exceptions=False
        )
        assert_command_succeeded(result1)

        # Second init with different template - should warn strongly
        result2 = runner.invoke(
            init, ["hyp-jumpstart-endpoint", temp_dir],
            input="n\n",  # Answer 'no' to re-initialize prompt
            catch_exceptions=False
        )

        # Verify strong warning is displayed
        assert_warning_displayed(result2, [
            "already initialized as",
            "highly unrecommended",
            "recommended path is create a new folder"
        ])
        assert_yes_no_prompt_displayed(result2)
        assert "aborting init" in result2.output.lower()


class TestSlurmClusterValidation:
    """Test config validation for slurm-cluster template."""

    def test_validate_valid_config_succeeds(self, temp_dir, runner):
        """Test that validation passes for valid config (Requirement 9.4)."""
        # Initialize slurm-cluster template
        result1 = runner.invoke(
            init, ["slurm-cluster", temp_dir], catch_exceptions=False
        )
        assert_command_succeeded(result1)

        # Update config with valid values - build complete config structure
        config_path = Path(temp_dir) / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Set valid values - ensure all required fields are populated
        config["cluster_name"] = "my-valid-cluster"
        config["vpc_config"] = {
            "subnets": ["subnet-12345678"],
            "security_group_ids": ["sg-12345678"]
        }

        # Set up instance groups with valid ARNs and S3 URIs
        config["instance_groups"] = [
            {
                "instance_group_name": "controller",
                "instance_type": "ml.t3.medium",
                "instance_count": 1,
                "execution_role": "arn:aws:iam::123456789012:role/SageMakerRole",
                "lifecycle_config": {
                    "source_s3_uri": "s3://my-bucket/lifecycle-scripts",
                    "on_create": "on_create.sh"
                }
            },
            {
                "instance_group_name": "worker",
                "instance_type": "ml.p4d.24xlarge",
                "instance_count": 2,
                "execution_role": "arn:aws:iam::123456789012:role/SageMakerRole",
                "lifecycle_config": {
                    "source_s3_uri": "s3://my-bucket/lifecycle-scripts",
                    "on_create": "on_create.sh"
                }
            }
        ]

        # Set up slurm_config
        config["slurm_config"] = {
            "controller_group": "controller",
            "worker_groups": [
                {
                    "instance_group_name": "worker",
                    "partition_name": "dev"
                }
            ]
        }

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Validate
        with change_directory(temp_dir):
            result2 = runner.invoke(validate, [], catch_exceptions=False)

        assert_command_succeeded(result2)
        assert_success_message_displayed(result2, ["✔️", "valid"])

    def test_validate_invalid_cluster_name_fails(self, temp_dir, runner):
        """Test that validation fails for invalid cluster name (Requirement 9.5)."""
        # Initialize slurm-cluster template
        result1 = runner.invoke(
            init, ["slurm-cluster", temp_dir], catch_exceptions=False
        )
        assert_command_succeeded(result1)

        # Set invalid cluster name (starts with hyphen)
        config_path = Path(temp_dir) / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        config["cluster_name"] = "-invalid-cluster-name"

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Validate should fail
        with change_directory(temp_dir):
            result2 = runner.invoke(validate, [], catch_exceptions=False)

        assert_command_failed_with_helpful_error(result2, ["cluster_name"])

    def test_validate_invalid_instance_group_structure_fails(self, temp_dir, runner):
        """Test that validation fails for invalid instance_groups structure (Requirement 9.1)."""
        # Initialize slurm-cluster template
        result1 = runner.invoke(
            init, ["slurm-cluster", temp_dir], catch_exceptions=False
        )
        assert_command_succeeded(result1)

        # Create a config with invalid instance_groups structure (missing required fields)
        config_path = Path(temp_dir) / "config.yaml"
        config = {
            "template": "slurm-cluster",
            "version": "1.0",
            "cluster_name": "test-cluster",
            "node_recovery": "Automatic",
            # instance_groups with invalid structure - missing required instance_group_name
            "instance_groups": [
                {
                    "instance_type": "ml.t3.medium",
                    "instance_count": 1,
                    # instance_group_name is missing - this is required
                }
            ],
            "vpc_config": {
                "subnets": ["subnet-12345678"],
                "security_group_ids": ["sg-12345678"]
            },
            "slurm_config": {
                "controller_group": "controller",
                "worker_groups": [{"instance_group_name": "worker", "partition_name": "dev"}]
            }
        }

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Validate should fail due to missing instance_group_name
        with change_directory(temp_dir):
            result2 = runner.invoke(validate, [], catch_exceptions=False)

        assert_command_failed_with_helpful_error(result2, ["instance_group_name"])

    def test_validate_invalid_slurm_config_reference_fails(self, temp_dir, runner):
        """Test that validation fails when slurm_config references non-existent instance group."""
        # Initialize slurm-cluster template
        result1 = runner.invoke(
            init, ["slurm-cluster", temp_dir], catch_exceptions=False
        )
        assert_command_succeeded(result1)

        # Create a config with invalid controller_group reference
        config_path = Path(temp_dir) / "config.yaml"
        config = {
            "template": "slurm-cluster",
            "version": "1.0",
            "cluster_name": "test-cluster",
            "node_recovery": "Automatic",
            "instance_groups": [
                {
                    "instance_group_name": "controller",
                    "instance_type": "ml.t3.medium",
                    "instance_count": 1,
                    "execution_role": "arn:aws:iam::123456789012:role/SageMakerRole",
                    "lifecycle_config": {
                        "source_s3_uri": "s3://my-bucket/scripts",
                        "on_create": "on_create.sh"
                    }
                }
            ],
            "vpc_config": {
                "subnets": ["subnet-12345678"],
                "security_group_ids": ["sg-12345678"]
            },
            "slurm_config": {
                "controller_group": "nonexistent-group",  # Invalid reference
                "worker_groups": [{"instance_group_name": "worker", "partition_name": "dev"}]
            }
        }

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Validate should fail
        with change_directory(temp_dir):
            result2 = runner.invoke(validate, [], catch_exceptions=False)

        assert_command_failed_with_helpful_error(result2, ["controller_group"])


class TestSlurmClusterReset:
    """Test reset command for slurm-cluster template."""

    def test_reset_clears_config_to_defaults(self, temp_dir, runner):
        """Test that reset command clears config back to default values."""
        # Initialize slurm-cluster template
        result1 = runner.invoke(
            init, ["slurm-cluster", temp_dir], catch_exceptions=False
        )
        assert_command_succeeded(result1)

        # Modify config
        config_path = Path(temp_dir) / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        config["cluster_name"] = "my-custom-cluster"
        config["node_recovery"] = "None"

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Verify config has custom values
        with open(config_path, "r") as f:
            modified_config = yaml.safe_load(f)
        assert modified_config["cluster_name"] == "my-custom-cluster"

        # Reset config
        with change_directory(temp_dir):
            result2 = runner.invoke(reset, [], catch_exceptions=False)

        assert_command_succeeded(result2)

        # Verify config was reset (template should remain)
        with open(config_path, "r") as f:
            reset_config = yaml.safe_load(f)

        assert reset_config.get("template") == "slurm-cluster", "Template should be preserved"
        # Cluster name should be reset to default
        assert reset_config.get("cluster_name") != "my-custom-cluster", "cluster_name should be reset"


class TestSlurmClusterConfigStructure:
    """Test the structure of generated slurm-cluster config."""

    def test_config_has_all_required_keys(self, temp_dir, runner):
        """Test that generated config has all required top-level keys."""
        result = runner.invoke(
            init, ["slurm-cluster", temp_dir], catch_exceptions=False
        )
        assert_command_succeeded(result)

        config_path = Path(temp_dir) / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Verify all required keys exist (values may be None as placeholders)
        required_keys = ["template", "version", "cluster_name", "instance_groups",
                        "vpc_config", "node_recovery", "slurm_config"]
        for key in required_keys:
            assert key in config, f"{key} should be present in config"

    def test_template_and_version_are_set(self, temp_dir, runner):
        """Test that template and version fields are properly set."""
        result = runner.invoke(
            init, ["slurm-cluster", temp_dir], catch_exceptions=False
        )
        assert_command_succeeded(result)

        config_path = Path(temp_dir) / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        assert config.get("template") == "slurm-cluster", "Template should be slurm-cluster"
        version = config.get("version")
        assert version == "1.0" or version == 1.0, f"Version should be 1.0, got {version}"

    def test_node_recovery_defaults_to_automatic(self, temp_dir, runner):
        """Test that node_recovery defaults to Automatic."""
        result = runner.invoke(
            init, ["slurm-cluster", temp_dir], catch_exceptions=False
        )
        assert_command_succeeded(result)

        config_path = Path(temp_dir) / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        assert config.get("node_recovery") == "Automatic", "node_recovery should default to Automatic"

    def test_populated_instance_groups_have_required_fields(self, temp_dir, runner):
        """Test that when instance_groups are populated, they have required fields."""
        result = runner.invoke(
            init, ["slurm-cluster", temp_dir], catch_exceptions=False
        )
        assert_command_succeeded(result)

        config_path = Path(temp_dir) / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # If instance_groups is populated (not None), verify structure
        if config.get("instance_groups") is not None and isinstance(config["instance_groups"], list):
            for ig in config["instance_groups"]:
                assert "instance_group_name" in ig, "instance_group_name is required"
                assert "instance_type" in ig, "instance_type is required"
                assert "instance_count" in ig, "instance_count is required"
                assert "execution_role" in ig, "execution_role is required"
                assert "lifecycle_config" in ig, "lifecycle_config is required"
                assert "source_s3_uri" in ig["lifecycle_config"], "source_s3_uri is required"
                assert "on_create" in ig["lifecycle_config"], "on_create is required"

    def test_populated_slurm_config_has_required_fields(self, temp_dir, runner):
        """Test that when slurm_config is populated, it has required fields."""
        result = runner.invoke(
            init, ["slurm-cluster", temp_dir], catch_exceptions=False
        )
        assert_command_succeeded(result)

        config_path = Path(temp_dir) / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # If slurm_config is populated (not None), verify structure
        slurm_config = config.get("slurm_config")
        if slurm_config is not None:
            assert "controller_group" in slurm_config, "controller_group is required"
            assert "worker_groups" in slurm_config, "worker_groups is required"

            # Verify worker_groups structure if populated
            if slurm_config.get("worker_groups") is not None:
                for wg in slurm_config["worker_groups"]:
                    assert "instance_group_name" in wg, "instance_group_name is required in worker_groups"
                    assert "partition_name" in wg, "partition_name is required in worker_groups"
