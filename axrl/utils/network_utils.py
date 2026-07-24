"""Network utility functions for port allocation and IP address detection.

This module provides utilities for:
- Finding available network ports with proper socket handling
- Detecting the host's IP address (IPv4/IPv6)
- Getting the default network interface

Key features of get_available_port():
- Uses context managers to ensure sockets are properly closed
- Applies SO_REUSEADDR to allow faster port reuse after TIME_WAIT
- Retries on failure to handle transient port conflicts
- Falls back to IPv6 if IPv4 fails
"""

import logging
import socket
import subprocess

logger = logging.getLogger(__name__)


def get_default_network_interface() -> str:
    """Get the default network interface name.

    Returns:
        The name of the default network interface (e.g., 'eth0', 'wlp2s0').

    Raises:
        RuntimeError: If the default network interface cannot be determined.
    """
    try:
        result = subprocess.check_output(["/usr/sbin/ip", "route", "show", "default"]).decode().strip()
        logger.debug(f"IP route show default output: {result}")
        # Example output: "default via 192.168.1.1 dev wlp2s0 proto dhcp metric 600"
        parts = result.split()
        if "dev" in parts:
            return parts[parts.index("dev") + 1]
    except Exception:
        logger.exception("Error getting default network interface.")
    raise RuntimeError("Could not determine default network interface.")


def get_ip() -> str:
    """Get the IP address of this host.

    Attempts IPv4 first, then falls back to IPv6 if IPv4 fails.

    Returns:
        The IP address as a string.

    Raises:
        RuntimeError: If no network connection is available.
    """
    # Try IPv4 first
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        logger.warning("Failed to get IPv4 address, trying IPv6.")

    # Fall back to IPv6
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as s:
            s.connect(("2001:4860:4860::8888", 80))
            return s.getsockname()[0]
    except Exception:
        logger.warning("Failed to get IPv6 address.")

    raise RuntimeError("No network connection available")


def get_available_port(max_attempts: int = 10) -> int:
    """Get an available port for binding.

    Uses SO_REUSEADDR to allow faster port reuse after connections in TIME_WAIT state.
    The function binds to port 0 to let the OS assign a free port, then verifies
    availability by calling listen().

    Args:
        max_attempts: Maximum number of retry attempts for IPv4 before falling back to IPv6.

    Returns:
        An available port number.

    Raises:
        RuntimeError: If no available port can be found after all attempts.
    """
    for attempt in range(max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("", 0))
                port = s.getsockname()[1]
                # Verify the port is truly available by trying to listen
                s.listen(1)
                return port
        except OSError as e:
            if attempt < max_attempts - 1:
                logger.debug(f"Port allocation attempt {attempt + 1} failed: {e}, retrying...")
            else:
                logger.warning("Failed to get available IPv4 port, trying IPv6.")

    # Fall back to IPv6
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", 0))
            s.listen(1)
            return s.getsockname()[1]
    except Exception:
        logger.warning("Failed to get available IPv6 port.")

    raise RuntimeError("No available port found")


if __name__ == "__main__":
    from axrl.utils import setup_logger

    setup_logger(level="debug")
    logger.info(f"Default network interface: {get_default_network_interface()}")
    logger.info(f"IP addresses: {get_ip()}")
    logger.info(f"Available port: {get_available_port()}")
