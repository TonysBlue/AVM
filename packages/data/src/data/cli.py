from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from model import StreamingLatentLoop
from runtime.config import load_config

from data.capture import (
    CaptureLedger,
    ExpertCaptureSession,
    speech_signal_from_base64,
)
from data.codec_targets import encode_target_speech
from data.curation import (
    audit_canary_data,
    build_canary_manifest,
    build_canary_text,
    check_readiness,
    fetch_canary_data,
    prepare_canary_data,
    select_canary_voices,
    synthesize_canary,
)
from data.curation.prepare import codec_client
from data.overfit import SpeechOverfitDataset
from data.ray import generate_synthetic_with_ray, write_ray_report
from data.replay import replay_episode, replay_ledger
from data.speech_import import import_speech_manifest
from data.synthetic import SyntheticEpisodeDataset
from data.webdataset import EpisodeShardReader, write_episode_shards


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="data")
    subparsers = parser.add_subparsers(dest="command", required=True)
    config_commands = {
        "generate-data",
        "build-overfit-data",
        "validate-data",
        "encode-speech",
        "import-speech",
        "inspect-model",
    }
    for name in config_commands:
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--set", action="append", default=[], dest="overrides")
    subparsers.choices["generate-data"].add_argument("--output")
    subparsers.choices["generate-data"].add_argument("--ray", action="store_true")
    subparsers.choices["build-overfit-data"].add_argument("--output", required=True)
    subparsers.choices["validate-data"].add_argument("--shards")
    subparsers.choices["encode-speech"].add_argument("--shards")
    subparsers.choices["encode-speech"].add_argument("--output", required=True)
    subparsers.choices["encode-speech"].add_argument("--native-source-lock")
    subparsers.choices["import-speech"].add_argument("--manifest", required=True)
    subparsers.choices["import-speech"].add_argument("--output", required=True)
    subparsers.choices["inspect-model"].add_argument("--report")
    native_assets = subparsers.add_parser("inspect-native-assets")
    native_assets.add_argument("--source-lock", required=True)
    native_assets.add_argument("--acoustic-only", action="store_true")
    native_speech = subparsers.add_parser("qualify-native-speech")
    native_speech.add_argument("--source-lock", required=True)
    native_speech.add_argument("--corpus", required=True)
    native_speech.add_argument("--output", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--config", required=True)
    capture.add_argument("--harness-socket", required=True)
    capture.add_argument("--scenario", required=True)
    capture.add_argument("--session-id", required=True)
    capture.add_argument("--lineage-id", required=True)
    capture.add_argument("--task-id", required=True)
    capture.add_argument("--split", choices=("train", "validation", "test"), default="train")
    capture.add_argument("--ledger", required=True)
    capture.add_argument("--output", required=True, help="staging shard pattern")
    capture.add_argument("--seed", type=int, default=17)
    capture.add_argument("--input", default="-", help="JSONL expert events or - for stdin")
    replay = subparsers.add_parser("replay")
    replay.add_argument("--harness-socket", required=True)
    replay_source = replay.add_mutually_exclusive_group(required=True)
    replay_source.add_argument("--ledger")
    replay_source.add_argument("--shards")
    replay.add_argument("--config")
    replay.add_argument("--episode-id")
    replay.add_argument("--scenario", required=True)
    replay.add_argument("--session-id", required=True)
    replay.add_argument("--seed", type=int, default=17)
    replay.add_argument("--non-realtime", action="store_true")
    readiness = subparsers.add_parser("check-readiness")
    readiness.add_argument("--config", required=True)
    readiness.add_argument("--root")
    prepare = subparsers.add_parser("prepare-canary-data")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--root")
    prepare.add_argument("--lock")
    prepare.add_argument("--download", action="store_true")
    prepare.add_argument("--extract", action="store_true")
    prepare.add_argument("--library")
    prepare.add_argument("--synth-command")
    prepare.add_argument("--asr-command")
    prepare.add_argument("--model-sha256")
    prepare.add_argument("--normalize-command")
    prepare.add_argument("--video-command")
    prepare.add_argument("--native-source-lock")
    prepare.add_argument("--encode", action="store_true")
    prepare.add_argument("--native-report-dir")
    prepare.add_argument("--fixture", action="store_true")
    for name in (
        "fetch-canary-data",
        "select-canary-voices",
        "build-canary-text",
        "synthesize-canary",
        "build-canary-manifest",
        "audit-canary-data",
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--root")
        command.add_argument("--fixture", action="store_true")
    subparsers.choices["fetch-canary-data"].add_argument("--lock")
    subparsers.choices["fetch-canary-data"].add_argument("--download", action="store_true")
    subparsers.choices["fetch-canary-data"].add_argument("--extract", action="store_true")
    subparsers.choices["select-canary-voices"].add_argument("--library")
    subparsers.choices["build-canary-text"].add_argument("--seed", type=int, default=17)
    subparsers.choices["synthesize-canary"].add_argument("--synth-command")
    subparsers.choices["synthesize-canary"].add_argument("--asr-command")
    subparsers.choices["synthesize-canary"].add_argument("--model-sha256")
    subparsers.choices["build-canary-manifest"].add_argument("--normalize-command")
    subparsers.choices["build-canary-manifest"].add_argument("--video-command")
    subparsers.choices["audit-canary-data"].add_argument("--native-report")
    args = parser.parse_args(argv)
    if args.command in {"inspect-native-assets", "qualify-native-speech"}:
        try:
            if args.command == "inspect-native-assets":
                from data.curation.native_assets import audit_native_assets

                report = audit_native_assets(args.source_lock, model_weights=not args.acoustic_only)
            else:
                from data.curation.native_speech import qualify_native_speech

                report = qualify_native_speech(args.source_lock, args.corpus, args.output)
        except (OSError, ValueError, RuntimeError) as error:
            print(json.dumps({"passed": False, "error": str(error)}), file=sys.stderr)
            return 2
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 2
    if args.command == "capture":
        from harness.transport.control import HarnessControlClient

        config = load_config(args.config)
        client = HarnessControlClient(args.harness_socket)
        harness_identity = client.identity()
        ledger = CaptureLedger(args.ledger, session_id=args.session_id, lineage_id=args.lineage_id)
        session = ExpertCaptureSession(client, ledger, seed=args.seed, scenario_id=args.scenario)
        source = sys.stdin
        try:
            session.start()
            source = (
                sys.stdin
                if args.input == "-"
                else Path(args.input).expanduser().open(encoding="utf-8")
            )
            for line in source:
                if not line.strip():
                    continue
                event = json.loads(line)
                speech = speech_signal_from_base64(
                    str(event.get("speech_pcm_b64", "")),
                )
                session.step(speech)
            episode = session.finish(
                {
                    "codec_id": config.data.codec_id,
                    "codec_weight_hash": config.data.codec_weight_hash,
                    "codec_revision": config.data.codec_revision,
                    **harness_identity,
                    "source": "expert-capture",
                    "source_license": "internal-consented-capture",
                    "redistribution_allowed": False,
                    "split": args.split,
                    "task_id": args.task_id,
                    "session_id_hash": hashlib.sha256(args.session_id.encode()).hexdigest(),
                    "device_id_hash": hashlib.sha256(
                        harness_identity["environment_id"].encode()
                    ).hexdigest(),
                }
            )
            write_episode_shards([episode], args.output)
        finally:
            if args.input != "-":
                source.close()
            session.close()
        print(
            json.dumps(
                {"session_id": args.session_id, "units": len(ledger.records), "sealed": True}
            )
        )
    elif args.command == "replay":
        from harness.transport.control import HarnessControlClient

        client = HarnessControlClient(args.harness_socket)

        def print_replay_event(event) -> None:
            print(
                json.dumps(
                    {
                        "unit_index": event.unit_index,
                        "receipt_accepted": event.receipt_accepted,
                        "execution_latency_ms": event.execution_latency_ms,
                        "observation_unit_index": event.observation_unit_index,
                        "elapsed_ms": event.elapsed_ms,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )

        replay_args = {
            "harness": client,
            "scenario_id": args.scenario,
            "seed": args.seed,
            "session_id": args.session_id,
            "realtime": not args.non_realtime,
            "step": False,
            "on_event": print_replay_event,
        }
        if args.ledger:
            events = replay_ledger(args.ledger, **replay_args)
        else:
            if not args.config:
                raise ValueError("--config is required with --shards")
            config = load_config(args.config)
            events = replay_episode(
                args.shards,
                data_config=config.data,
                model_config=config.model,
                episode_id=args.episode_id,
                **replay_args,
            )
        print(
            json.dumps(
                {
                    "session_id": args.session_id,
                    "units": len(events),
                    "realtime": not args.non_realtime,
                }
            )
        )
    elif args.command == "check-readiness":
        config = load_config(args.config)
        root = args.root or config.runtime.data_root
        print(json.dumps(check_readiness(root, config=config), indent=2))
    elif args.command == "prepare-canary-data":
        config = load_config(args.config)
        root = args.root or config.runtime.data_root
        report = prepare_canary_data(
            root,
            config=config,
            fixture=args.fixture,
            lock_path=args.lock,
            download=args.download,
            extract=args.extract,
            library=args.library,
            synth_command=args.synth_command,
            asr_command=args.asr_command,
            model_sha256=args.model_sha256,
            normalize_command=args.normalize_command,
            video_command=args.video_command,
            native_source_lock=args.native_source_lock,
            encode=args.encode,
            native_report_dir=args.native_report_dir,
        )
        print(json.dumps(report, indent=2, default=str))
    elif args.command in {
        "fetch-canary-data",
        "select-canary-voices",
        "build-canary-text",
        "synthesize-canary",
        "build-canary-manifest",
        "audit-canary-data",
    }:
        config = load_config(args.config)
        root = args.root or config.runtime.data_root
        if args.command == "fetch-canary-data":
            report = fetch_canary_data(
                root,
                fixture=args.fixture,
                lock_path=args.lock,
                download=args.download,
                extract=args.extract,
            )
        elif args.command == "select-canary-voices":
            report = select_canary_voices(root, library=args.library, fixture=args.fixture)
        elif args.command == "build-canary-text":
            report = build_canary_text(root, fixture=args.fixture, seed=args.seed)
        elif args.command == "synthesize-canary":
            report = synthesize_canary(
                root,
                fixture=args.fixture,
                synth_command=args.synth_command,
                asr_command=args.asr_command,
                model_sha256=args.model_sha256,
            )
        elif args.command == "build-canary-manifest":
            report = build_canary_manifest(
                root,
                fixture=args.fixture,
                normalize_command=args.normalize_command,
                video_command=args.video_command,
            )
        else:
            report = audit_canary_data(root, fixture=args.fixture, native_report=args.native_report)
        print(json.dumps(report, indent=2, default=str))
    else:
        config = load_config(args.config, args.overrides)
        if args.command == "inspect-model":
            model = StreamingLatentLoop(config.model, load_native=False)
            report = {
                "parameters": model.parameter_count(),
                "tokens_per_unit": config.model.tokens_per_unit,
                "perceiver_slots": config.model.perceiver_slots,
                "perceiver_layers": config.model.perceiver_layers,
                "jepa_layers": config.model.jepa_layers,
                "max_kv_tokens": config.model.kv_units * config.model.perceiver_slots,
                "weights_loaded": False,
            }
        elif args.command == "generate-data":
            output = args.output or str(config.runtime.data_path() / "generated" / "train-%06d.tar")
            manifest = (
                generate_synthetic_with_ray(config, output)
                if args.ray
                else write_episode_shards(
                    SyntheticEpisodeDataset(config.data, config.model), output
                )
            )
            if args.ray:
                write_ray_report(
                    config.runtime.data_path() / "generated" / "ray-report.json", manifest
                )
            report = {"episodes": len(manifest), "output": output}
        elif args.command == "build-overfit-data":
            manifest = write_episode_shards(
                SpeechOverfitDataset(config.data, config.model), args.output
            )
            report = {"episodes": len(manifest), "output": args.output}
        elif args.command == "validate-data":
            source = args.shards or config.data.shards
            if not source:
                raise ValueError("provide --shards or configure data.shards")
            episodes = units = 0
            for episode in EpisodeShardReader(source, config.data, config.model):
                episodes += 1
                units += len(episode.units)
            report = {"episodes": episodes, "units": units}
        elif args.command == "import-speech":
            manifest = write_episode_shards(
                import_speech_manifest(args.manifest, config.data, config.model), args.output
            )
            report = {"episodes": len(manifest), "output": args.output}
        elif args.command == "encode-speech":
            source = args.shards or config.data.shards
            if not source:
                raise ValueError("provide --shards or configure data.shards")
            with codec_client(config, args.native_source_lock) as client:
                manifest = write_episode_shards(
                    encode_target_speech(
                        EpisodeShardReader(
                            source,
                            config.data,
                            config.model,
                            require_encoded_speech=False,
                            validate_manifest=False,
                        ),
                        client,
                    ),
                    args.output,
                )
            report = {"episodes": len(manifest), "output": args.output}
        else:
            raise ValueError(f"unknown data command: {args.command}")
        print(json.dumps(report, indent=2, default=str))
    return 0
