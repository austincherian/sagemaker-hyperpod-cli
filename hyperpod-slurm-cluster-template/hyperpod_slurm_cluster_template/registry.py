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
"""Registry for Slurm cluster template versions."""

from typing import Dict, Type

from pydantic import BaseModel

from .v1_0 import model as v1_0_model
from .v1_0.template import CONFIG_TEMPLATE_CONTENT as v1_0_config_template
from .v1_0.template import PROVISIONING_PARAMS_TEMPLATE_CONTENT as v1_0_provisioning_template

# Direct version-to-model mapping
SCHEMA_REGISTRY: Dict[str, Type[BaseModel]] = {
    "1.0": v1_0_model.SlurmClusterTemplateConfig,
}

# Template registry for config.yaml scaffolding
TEMPLATE_REGISTRY: Dict[str, str] = {
    "1.0": v1_0_config_template,
}

# Provisioning parameters template registry
PROVISIONING_PARAMS_REGISTRY: Dict[str, str] = {
    "1.0": v1_0_provisioning_template,
}
