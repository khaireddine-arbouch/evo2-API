<<<<<<< HEAD
#!/usr/bin/env python3
"""
Health check script for Evo2 Variant Pathogenicity Prediction API.

This script tests the API endpoint to verify it's operational.
"""

import json
import os
import sys
from typing import Optional

import requests


def check_api_health(
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = 120,
) -> tuple[bool, str, int]:
    """
    Check if the API endpoint is healthy and responding.

    Args:
        endpoint: API endpoint URL
        api_key: Optional API key for authentication
        timeout: Request timeout in seconds

    Returns:
        Tuple of (is_healthy, message, status_code)
    """
    if not endpoint:
        endpoint = os.environ.get("MODAL_API_ENDPOINT")
        if not endpoint:
            return False, "API endpoint not configured", 0

    # Test request payload (BRCA1 variant example)
    test_payload = {
        "variant_position": 43119628,
        "alternative": "G",
        "genome": "hg38",
        "chromosome": "chr17",
        "mutation_type": "SNV",
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        print(f"Checking API endpoint: {endpoint}")
        print(f"Test payload: {json.dumps(test_payload, indent=2)}")

        response = requests.post(
            endpoint,
            json=test_payload,
            headers=headers,
            timeout=timeout,
        )

        status_code = response.status_code
        print(f"HTTP Status Code: {status_code}")

        if status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            return True, "API is operational", status_code

        elif status_code in (401, 403):
            return False, "Authentication failed - check API key", status_code

        elif status_code >= 500:
            error_detail = response.text[:500] if response.text else "No error details"
            return False, f"Server error: {error_detail}", status_code

        else:
            error_detail = response.text[:500] if response.text else "No error details"
            return False, f"Unexpected status: {error_detail}", status_code

    except requests.exceptions.Timeout:
        return False, "Request timed out - API may be slow or unreachable", 0

    except requests.exceptions.ConnectionError:
        return False, "Connection failed - API endpoint is unreachable", 0

    except requests.exceptions.RequestException as e:
        return False, f"Request failed: {str(e)}", 0

    except Exception as e:
        return False, f"Unexpected error: {str(e)}", 0


def main():
    """Main entry point for health check script."""
    endpoint = os.environ.get("MODAL_API_ENDPOINT")
    api_key = os.environ.get("MODAL_API_KEY")

    if not endpoint:
        print("WARNING: MODAL_API_ENDPOINT environment variable not set")
        print("Skipping health check.")
        sys.exit(0)

    is_healthy, message, status_code = check_api_health(endpoint, api_key)

    if is_healthy:
        print(f"SUCCESS: {message}")
        sys.exit(0)
    else:
        print(f"FAILED: {message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
=======
#!/usr/bin/env python3
"""
Health check script for Evo2 Variant Pathogenicity Prediction API.

This script tests the API endpoint to verify it's operational.
"""

import json
import os
import sys
from typing import Optional

import requests


def check_api_health(
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = 120,
) -> tuple[bool, str, int]:
    """
    Check if the API endpoint is healthy and responding.

    Args:
        endpoint: API endpoint URL
        api_key: Optional API key for authentication
        timeout: Request timeout in seconds

    Returns:
        Tuple of (is_healthy, message, status_code)
    """
    if not endpoint:
        endpoint = os.environ.get("MODAL_API_ENDPOINT")
        if not endpoint:
            return False, "API endpoint not configured", 0

    # Test request payload (BRCA1 variant example)
    test_payload = {
        "variant_position": 43119628,
        "alternative": "G",
        "genome": "hg38",
        "chromosome": "chr17",
        "mutation_type": "SNV",
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        print(f"Checking API endpoint: {endpoint}")
        print(f"Test payload: {json.dumps(test_payload, indent=2)}")

        response = requests.post(
            endpoint,
            json=test_payload,
            headers=headers,
            timeout=timeout,
        )

        status_code = response.status_code
        print(f"HTTP Status Code: {status_code}")

        if status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            return True, "API is operational", status_code

        elif status_code in (401, 403):
            return False, "Authentication failed - check API key", status_code

        elif status_code >= 500:
            error_detail = response.text[:500] if response.text else "No error details"
            return False, f"Server error: {error_detail}", status_code

        else:
            error_detail = response.text[:500] if response.text else "No error details"
            return False, f"Unexpected status: {error_detail}", status_code

    except requests.exceptions.Timeout:
        return False, "Request timed out - API may be slow or unreachable", 0

    except requests.exceptions.ConnectionError:
        return False, "Connection failed - API endpoint is unreachable", 0

    except requests.exceptions.RequestException as e:
        return False, f"Request failed: {str(e)}", 0

    except Exception as e:
        return False, f"Unexpected error: {str(e)}", 0


def main():
    """Main entry point for health check script."""
    endpoint = os.environ.get("MODAL_API_ENDPOINT")
    api_key = os.environ.get("MODAL_API_KEY")

    if not endpoint:
        print("WARNING: MODAL_API_ENDPOINT environment variable not set")
        print("Skipping health check.")
        sys.exit(0)

    is_healthy, message, status_code = check_api_health(endpoint, api_key)

    if is_healthy:
        print(f"SUCCESS: {message}")
        sys.exit(0)
    else:
        print(f"FAILED: {message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
>>>>>>> 8270dd25e630f73690688d5065950dfed596e9cf
