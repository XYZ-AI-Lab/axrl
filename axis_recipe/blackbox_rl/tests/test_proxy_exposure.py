from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

import pytest

from axis_recipe.blackbox_rl.config import OpenAIProxyConfig, OpenAIProxyExposureConfig, OpenAIProxyTunnelConfig
from axis_recipe.blackbox_rl.train_blackbox_rl import _resolve_proxy_exposure
from axrl.utils.tunnel import Tunnel, allow_out_for_base_url, is_public_routable_host

if TYPE_CHECKING:
    from axrl.openai_proxy import OpenAIProxyServer


class FakeServer:
    host = "127.0.0.1"
    port = 8080


def test_proxy_exposure_derives_allow_out_from_exposed_url() -> None:
    proxy_config = OpenAIProxyConfig(
        host="127.0.0.1",
        exposure=OpenAIProxyExposureConfig(exposed_base_url="https://proxy.example.com/{port}", allow_out=["extra.example.com"], tunnel=None),
    )

    base_url, allow_out, tunnel = asyncio.run(_resolve_proxy_exposure(proxy_config, cast("OpenAIProxyServer", FakeServer())))

    assert base_url == "https://proxy.example.com/8080"
    assert allow_out == ("proxy.example.com", "extra.example.com")
    assert tunnel is None


def test_proxy_exposure_defaults_to_cloudflare_tunnel(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_start(
        config: OpenAIProxyTunnelConfig,
        *,
        template_vars: dict[str, object],
        drain_task_name: str,
    ) -> FakeTunnel:
        assert config.command == ["cloudflared", "tunnel", "--url", "http://127.0.0.1:{port}"]
        assert template_vars == {"base_url": "http://127.0.0.1:8080", "host": "127.0.0.1", "port": 8080}
        assert drain_task_name == "blackbox-openai-proxy-tunnel-drain"
        return FakeTunnel(base_url="https://abc.trycloudflare.com")

    monkeypatch.setattr(Tunnel, "start", fake_start)
    base_url, allow_out, tunnel = asyncio.run(_resolve_proxy_exposure(OpenAIProxyConfig(host="127.0.0.1"), cast("OpenAIProxyServer", FakeServer())))

    assert base_url == "https://abc.trycloudflare.com"
    assert allow_out == ("abc.trycloudflare.com",)
    assert isinstance(tunnel, FakeTunnel)


def test_proxy_tunnel_reports_missing_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("axrl.utils.tunnel.shutil.which", lambda _name: None)
    config = OpenAIProxyTunnelConfig(command=["missing-cloudflared", "tunnel"])

    with pytest.raises(FileNotFoundError, match="missing-cloudflared"):
        asyncio.run(
            Tunnel.start(
                config,
                template_vars={"base_url": "http://127.0.0.1:8080", "host": "127.0.0.1", "port": 8080},
            )
        )


def test_proxy_exposure_rejects_private_direct_host_without_exposure() -> None:
    proxy_config = OpenAIProxyConfig(host="127.0.0.1", exposure=OpenAIProxyExposureConfig(tunnel=None))

    with pytest.raises(ValueError, match="no E2B-routable exposure"):
        asyncio.run(_resolve_proxy_exposure(proxy_config, cast("OpenAIProxyServer", FakeServer())))


def test_proxy_exposure_rejects_broad_allow_out() -> None:
    with pytest.raises(ValueError, match="broad"):
        allow_out_for_base_url("https://proxy.example.com", ["0.0.0.0/0"])


def test_proxy_routable_host_check() -> None:
    assert is_public_routable_host("proxy.example.com")
    assert not is_public_routable_host("127.0.0.1")
    assert not is_public_routable_host("10.0.0.5")


class FakeTunnel:
    def __init__(self, *, base_url: str) -> None:
        self.base_url = base_url

    async def stop(self) -> None:
        return None
