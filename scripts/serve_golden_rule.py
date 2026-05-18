"""Serve a Golden Rule policy for BEHAVIOR-1K evaluation.

This script is the server-side counterpart to ``eval_golden_rule.py``.  It
creates a ``GoldenRulePolicyWrapper`` (instead of the standard
``B1KPolicyWrapper``) so that the policy is driven by a ground-truth skill
plan rather than a VLM reasoner.

Usage:
    # Serve with a GT plan loaded from demo annotations
    python serve_golden_rule.py \
        --policy.config pi0_b1k \
        --policy.dir /path/to/checkpoint \
        --task_name turning_on_radio \
        --demo_data_path /path/to/2025-challenge-demos \
        --demo_id 00000010 \
        --fine_grained_level 2

    # The evaluator (eval_golden_rule.py) connects to this server via
    # websocket.  The server autonomously advances through the skill plan
    # based on timeout or skill-completion detection.
"""

import dataclasses
import enum
import logging
import socket
from typing import Any

from omnigibson.learning.utils.network_utils import WebsocketPolicyServer
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.shared.golden_rule_policy import GoldenRulePolicyWrapper
from openpi.training import config as _config

logger = logging.getLogger("serve_golden_rule")


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    config: str
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_golden_rule script."""

    env: EnvMode = EnvMode.ALOHA_SIM

    default_prompt: str | None = None
    prompt_override: str | None = None

    dataset_root: str | None = "/scr/behavior/2025-challenge-demos"
    task_name: str | None = None

    port: int = 8000
    server_run_id: str | None = None
    server_token: str | None = None
    record: bool = False
    policy_backend: str = "auto"

    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    fine_grained_level: int = 2
    control_mode: str = "receeding_horizon"
    max_len: int = 32
    action_horizon: int = 5
    temporal_ensemble_max: int = 3

    # Golden rule specific
    skill_timeout_steps: int = 300

    # If provided, the server will load a GT plan from demo annotations and
    # inject it into the policy wrapper.  When None, the wrapper starts with
    # an empty plan and expects the client (evaluator) to set it later via
    # the websocket protocol (not yet implemented).
    demo_data_path: str | None = None
    demo_id: str | None = None


def _try_load_plan_loader(
    demo_data_path: str | None,
    task_name: str | None,
    demo_id: str | None,
) -> Any:
    """Attempt to create a GTPlanLoader from the BEHAVIOR-1K repo.

    This is a lazy import so that openpi-comet does not depend on
    OmniGibson at import time.
    """
    if demo_data_path is None or task_name is None or demo_id is None:
        return None

    try:
        from omnigibson.learning.gt_plan_loader import GTPlanLoader  # type: ignore[import-untyped]
    except Exception as exc:
        logger.warning(
            "Could not import GTPlanLoader from BEHAVIOR-1K (%s). "
            "The server will start without a GT plan.",
            exc,
        )
        return None

    try:
        loader = GTPlanLoader(
            demo_data_path=demo_data_path,
            task_name=task_name,
            demo_id=demo_id,
        )
        plan = loader.load_plan()
        if plan:
            logger.info(
                "Loaded GT plan with %d skills for task=%s demo=%s",
                len(plan),
                task_name,
                demo_id,
            )
            return loader
        else:
            logger.warning(
                "Empty GT plan for task=%s demo=%s",
                task_name,
                demo_id,
            )
            return None
    except Exception as exc:
        logger.warning("Failed to load GT plan: %s", exc)
        return None


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    return _policy_config.create_trained_policy(
        _config.get_config(args.policy.config),
        args.policy.dir,
        default_prompt=args.default_prompt,
        policy_backend=args.policy_backend,
    )


def main(args: Args) -> None:
    logging.info("Using task_name: %s", args.task_name)

    policy = create_policy(args)
    base_policy_metadata = dict(policy.metadata or {})

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    # Load GT plan if demo_data_path is provided
    plan_loader = _try_load_plan_loader(
        args.demo_data_path,
        args.task_name,
        args.demo_id,
    )

    policy = GoldenRulePolicyWrapper(
        policy=policy,
        task_name=args.task_name,
        plan_loader=plan_loader,
        control_mode=args.control_mode,
        max_len=args.max_len,
        action_horizon=args.action_horizon,
        skill_timeout_steps=args.skill_timeout_steps,
        fine_grained_level=args.fine_grained_level,
        temporal_ensemble_max=args.temporal_ensemble_max,
    )
    policy_metadata = {
        **base_policy_metadata,
        **policy.server_identity_metadata(),
        "server_run_id": args.server_run_id,
        "server_token": args.server_token,
        "golden_rule": True,
        "n_plan_skills": len(plan_loader) if plan_loader is not None else 0,
    }
    policy_metadata = {k: v for k, v in policy_metadata.items() if v is not None}

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        local_ip = "127.0.0.1"
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
    policy_metadata.update({
        "server_hostname": hostname,
        "server_ip": local_ip,
        "server_port": args.port,
    })

    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
