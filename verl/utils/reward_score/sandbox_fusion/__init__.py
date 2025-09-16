# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import traceback
import re

# from .utils import check_correctness
from .utils_common_evaluate import check_correctness_common_evaluate

"""
Verify code correctness using the Sandbox Fusion (https://github.com/bytedance/SandboxFusion).
You can either deploy the sandbox_fusion service yourself or use the
FaaS service provided by public cloud, eg: volcengine.com.
"""
logger = logging.getLogger(__name__)

def compute_score(sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, completion, test_cases, continuous=False, timeout=30):
    """
    Computes the code score using the remote sandbox API.

    Args:
        sandbox_fusion_url: The URL of the sandbox_fusion service, eg: "https://<your service endpoint>/run_code"

        completion: The completion string containing the code.
        test_cases: JSON string or dictionary containing "inputs" and "outputs".
        continuous: Whether to compute a continuous score (based on the first N test cases).
        timeout: Timeout for each test case.

    Returns:
        A tuple (score, metadata_list).
        score: Float score (0.0 to 1.0).
        metadata_list: List containing execution metadata for each test case.
    """
    # remove <think>.*</think> tags if they exist
    completion = re.sub(r'<think>.*?</think>', '', completion, flags=re.DOTALL).strip()
    solution = completion
    language_str = re.search(r'```(\w+)', completion)
    if language_str:
        language = language_str.group(1).strip()
    else:
        # Default to Python if no language is specified
        language = "python"
        
    try:
        if not isinstance(test_cases, dict):
            try:
                test_cases = json.loads(test_cases)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse test_cases JSON: {e}")
                return 0.0, [{"error": "Invalid test_cases JSON format"}]

        if not test_cases or "input" not in test_cases or "output" not in test_cases:
            logger.error("Invalid test_cases structure.")
            logger.error(f"{test_cases}")
            return 0.0, [{"error": "Invalid test_cases structure (missing inputs/outputs)"}]

        # Check all test cases
        # Note: The return value of check_correctness might need adaptation here
        # Assume check_correctness returns (results_list, metadata_list)
        # results_list contains True, False, or error codes (-1, -2, -3, etc.)
        res_list, metadata_list = check_correctness_common_evaluate(sandbox_fusion_url=sandbox_fusion_url, in_outs=test_cases, generation=solution, timeout=timeout, concurrent_semaphore=concurrent_semaphore, language=language)

        # Calculate score
        if not res_list:  # If there are no results (e.g., invalid input)
            return 0.0, metadata_list

        score = metadata_list[0].get("score", 0.0)  # Default to 0.0 if score not found
        final_metadata = metadata_list

    except Exception as e:
        # logger.error(f"Error during compute_score: {e}")
        # traceback.print_exc()
        score = 0.0
        # Try to return partial metadata if available, otherwise return error info
        final_metadata = metadata_list if "metadata_list" in locals() else [{"error": f"Unhandled exception: {e}"}]

    # Ensure float and list are returned
    return float(score), final_metadata if isinstance(final_metadata, list) else [final_metadata]
