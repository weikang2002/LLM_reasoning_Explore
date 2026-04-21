"""
CLI tool to check platform compatibility and show recommended settings.
"""

import sys
from interpretability.core.platform_utils import print_platform_info


def main():
    """Main entry point for platform check command."""
    print_platform_info()
    return 0


if __name__ == "__main__":
    sys.exit(main())
