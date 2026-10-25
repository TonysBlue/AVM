from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from harness.transport.control import HarnessControlServer


def _load_backend(config_path: Path, adapter_module: str | None = None):
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError("Harness service config must be a mapping")
    module_name = adapter_module or "harness.deployment.dialogue"
    module = __import__(module_name, fromlist=["create_backend"])
    if hasattr(module, "create_backend"):
        backend = module.create_backend(config)
        _validate_backend(backend, module_name)
        return backend
    raise RuntimeError("deployment module must expose create_backend(config)")


def _validate_backend(backend: object, adapter_module: str) -> None:
    required = (
        "environment_id",
        "environment_version",
        "start_lifetime_session",
        "apply",
        "close",
    )
    missing = [name for name in required if not hasattr(backend, name)]
    if missing:
        raise RuntimeError(f"deployment module {adapter_module!r} backend is missing: {missing}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness")
    parser.add_argument("command", choices=("serve", "converse"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-module", help="test-only deployment module override")
    parser.add_argument("--socket", help="override control socket path")
    parser.add_argument("--model-socket")
    parser.add_argument("--scenario")
    parser.add_argument("--session-id")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-units", type=int)
    args = parser.parse_args(argv)
    config = Path(args.config).expanduser()
    if not config.is_file():
        raise FileNotFoundError(config)
    raw = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Harness service config must be a mapping")
    socket_path = args.socket or str(raw.get("control_socket", raw.get("model_service_socket", "")))
    if not socket_path:
        raise ValueError("Harness service config requires control_socket")
    if args.command == "converse":
        from harness.conversation import converse
        from harness.transport.control import HarnessControlClient
        from harness.transport.model_client import UnixModelServiceClient

        model_socket = args.model_socket or str(raw.get("model_service_socket", ""))
        if not model_socket or not args.scenario or not args.session_id:
            parser.error("converse requires model socket, --scenario and --session-id")
        print(json.dumps(converse(
            HarnessControlClient(socket_path), UnixModelServiceClient(model_socket),
            scenario_id=args.scenario, session_id=args.session_id, seed=args.seed,
            max_units=args.max_units,
        )))
        return 0
    HarnessControlServer(
        lambda: _load_backend(config, args.adapter_module),
        socket_path,
        expected_environment_id=str(raw.get("environment_id", "")) or None,
        expected_environment_version=str(raw.get("environment_version", "")) or None,
        expected_protocol_version=str(raw.get("protocol_version", "avm-realtime-v2")),
    ).serve_forever()
    return 0
