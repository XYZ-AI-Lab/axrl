import os

from axrl.utils.network_utils import get_available_port, get_default_network_interface, get_ip


class Worker:
    """Base class for all workers."""

    def __init__(self) -> None:
        self.ip = self.get_ip()
        self.set_default_network_interface()

    def initialize(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    @staticmethod
    def get_cuda_visible_devices() -> str:
        devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        return devices

    def get_ip_and_first_cuda_visible_device(self) -> tuple[str, str]:
        ip = self.get_ip()
        cuda_visible_devices = self.get_cuda_visible_devices()
        first_device = cuda_visible_devices.split(",")[0] if cuda_visible_devices else ""
        return ip, first_device

    @staticmethod
    def set_cuda_visible_devices(cuda_visible_devices: str) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    @staticmethod
    def get_ip() -> str:
        return get_ip()

    @staticmethod
    def get_port() -> int:
        return get_available_port()

    @staticmethod
    def get_default_network_interface() -> str:
        return get_default_network_interface()

    @staticmethod
    def get_available_address() -> str:
        ip = Worker.get_ip()
        port = Worker.get_port()
        return f"{ip}:{port}"

    @staticmethod
    def set_default_network_interface() -> None:
        interface = Worker.get_default_network_interface()
        assert interface is not None, "No network interface detected"
        os.environ["GLOO_SOCKET_IFNAME"] = interface
