from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf
from runtime.config import ProjectConfig, load_config
from runtime.native_acoustics import NativeAcousticBackend, NativeAcousticServer
from runtime.native_assets import audit_native_assets, require_native_qualification

from model_service.service import ModelService
from model_service.transport.server import UnixModelServer


def _load_service_config(path: str | Path) -> tuple[ProjectConfig, str | None, str | None]:
    """Load a project config plus optional service-level socket/device settings."""
    config_path = Path(path).expanduser().resolve()
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
    if not isinstance(raw, dict) or "config" not in raw:
        return load_config(config_path), None, None
    project_path = Path(str(raw["config"])).expanduser()
    if not project_path.is_absolute():
        project_path = (config_path.parent / project_path).resolve()
    return (
        load_config(project_path),
        str(raw["socket"]) if raw.get("socket") else None,
        str(raw["device"]) if raw.get("device") else None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="model-service")
    parser.add_argument("command", choices=("serve", "serve-acoustics"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--socket")
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    config, service_socket, service_device = _load_service_config(args.config)
    socket_path = args.socket or service_socket
    if not socket_path:
        parser.error("--socket is required unless the service config provides socket")
    device = args.device or service_device or "cpu"
    if args.command == "serve-acoustics":
        if not config.model.native_source_lock:
            raise ValueError("native acoustic server requires a locked source asset")
        audit = audit_native_assets(config.model.native_source_lock, model_weights=False)
        require_native_qualification(audit, config.model.native_qualification)
        decoder = NativeAcousticBackend(audit, cuda_device=config.runtime.acoustic_cuda_device)
        NativeAcousticServer(decoder, socket_path).serve_forever()
        return 0
    UnixModelServer(
        ModelService(config, args.checkpoint, device), socket_path
    ).serve_forever()
    return 0
