from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from data.curation.common import (
    dataset_path,
    ensure_tree,
    sha256_file,
    stable_hash,
    write_json,
)
from data.curation.spec import (
    LANGUAGE_FRACTIONS,
    SPLIT_FRACTIONS,
    dataset_spec,
)

INTENTS = {
    "visual_description": 20,
    "event_tracking": 15,
    "audiovisual_grounding": 15,
    "shared_viewing": 15,
    "conversation_memory": 10,
    "clarification": 10,
    "interruption_recovery": 10,
    "everyday_dialogue": 5,
}

ZH_COPY = {
    'visual_description': (
        '请描述你看到的画面。',
        '我会结合画面来描述。',
        '我们可以继续观察细节。',
    ),
    'event_tracking': (
        '留意接下来发生的变化。',
        '我会持续观察。',
        '我们来回顾刚才的变化。',
    ),
    'audiovisual_grounding': (
        '结合画面和声音一起听看。',
        '我正在同时观察和聆听。',
        '我们可以讨论声音与画面的联系。',
    ),
    'shared_viewing': (
        '陪我一起看这段视频。',
        '好，我们边看边聊。',
        '你想聊刚才的哪个细节？',
    ),
    'conversation_memory': (
        '请记住我们刚才谈到的事情。',
        '我会结合之前的对话继续交流。',
        '我们可以回顾前面的内容。',
    ),
    'clarification': (
        '我说的是刚才那个。',
        '你指的是画面里的哪一处？',
        '请再描述一点，我继续听。',
    ),
    'interruption_recovery': (
        '先停一下，我补充一句。',
        '好的，你说。',
        '我们接着刚才的话题。',
    ),
    'everyday_dialogue': (
        '今天想和你聊聊天。',
        '好啊，你想从什么开始？',
        '我在听，你继续。',
    ),
}

_FORMAL_DURATION_COUNTS = {
    ("zh", "train"): (48, 19, 9),
    ("zh", "validation"): (6, 3, 1),
    ("zh", "test"): (6, 2, 2),
    ("en", "train"): (9, 4, 6),
    ("en", "validation"): (1, 1, 0),
    ("en", "test"): (2, 1, 0),
}
EN_COPY = {
    'visual_description': (
        'Describe what you see.',
        'Let us look at the scene together.',
        'We can discuss the details.',
    ),
    'event_tracking': (
        'Watch for changes.',
        'I will keep watching.',
        'Let us review what changed.',
    ),
    'audiovisual_grounding': (
        'Listen and watch together.',
        'I am following the sound and video.',
        'We can discuss their connection.',
    ),
    'shared_viewing': (
        'Watch this video with me.',
        'Sure, let us talk as we watch.',
        'Which detail would you like to discuss?',
    ),
    'conversation_memory': (
        'Remember what we discussed.',
        'I will keep our earlier conversation in mind.',
        'We can revisit it together.',
    ),
    'clarification': (
        'I meant that one earlier.',
        'Which part of the scene do you mean?',
        'Tell me a little more.',
    ),
    'interruption_recovery': (
        'Pause, I want to add something.',
        'Go ahead, I am listening.',
        'Let us return to our conversation.',
    ),
    'everyday_dialogue': (
        'I would like to chat.',
        'What would you like to talk about?',
        'I am listening.',
    ),
}


def _weighted_intents(count: int) -> list[str]:
    values: list[str] = []
    for intent, percent in INTENTS.items():
        values.extend([intent] * (count * percent // 100))
    while len(values) < count:
        values.append(list(INTENTS)[len(values) % len(INTENTS)])
    return values[:count]


def plan_recipe_sha256(plan: dict[str, Any]) -> str:
    return stable_hash(
        {key: value for key, value in plan.items() if key != "recipe_sha256"}
    )


def _split(index: int, count: int) -> str:
    if index < count * 8 // 10:
        return "train"
    return "validation" if index < count * 9 // 10 else "test"


def _duration_bounds(plan: dict[str, Any]) -> tuple[float, float]:
    return {
        "short": (4.0, 16.0),
        "medium": (16.0, 32.0),
        "long": (32.0, 60.0),
    }[str(plan["duration_class"])]


def _allocate_duration(plans: list[dict[str, Any]], target: float) -> None:
    bounds = [_duration_bounds(plan) for plan in plans]
    minimum = sum(lower for lower, _ in bounds)
    maximum = sum(upper for _, upper in bounds)
    if not minimum <= target <= maximum:
        raise ValueError(
            f"plan duration capacity [{minimum}, {maximum}] cannot cover {target} seconds"
        )
    capacities = [upper - lower for lower, upper in bounds]
    remaining = target - minimum
    total_capacity = sum(capacities)
    durations = [
        lower + (remaining * capacity / total_capacity if total_capacity else 0.0)
        for (lower, _), capacity in zip(bounds, capacities, strict=True)
    ]
    correction = target - sum(durations)
    durations[-1] += correction
    for plan, duration in zip(plans, durations, strict=True):
        plan["target_duration_seconds"] = round(duration, 6)


def _shift_video_duration(plans: list[dict[str, Any]], target: float) -> None:
    current = sum(
        float(plan["target_duration_seconds"])
        for plan in plans
        if plan["category"] == "video_task"
    )
    delta = target - current
    if abs(delta) < 1e-6:
        return
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for plan in plans:
        groups.setdefault((str(plan["language"]), str(plan["split"])), []).append(plan)
    for group in groups.values():
        video = [plan for plan in group if plan["category"] == "video_task"]
        dialogue = [plan for plan in group if plan["category"] == "synthetic_dialogue"]
        if delta > 0:
            receivers, donors = video, dialogue
        else:
            receivers, donors = dialogue, video
        receiver_capacity = sum(
            _duration_bounds(plan)[1] - float(plan["target_duration_seconds"]) for plan in receivers
        )
        donor_capacity = sum(
            float(plan["target_duration_seconds"]) - _duration_bounds(plan)[0] for plan in donors
        )
        moved = min(abs(delta), receiver_capacity, donor_capacity)
        if moved <= 0:
            continue
        left = moved
        for plan in receivers:
            capacity = _duration_bounds(plan)[1] - float(plan["target_duration_seconds"])
            increment = min(left, capacity)
            plan["target_duration_seconds"] = float(plan["target_duration_seconds"]) + increment
            left -= increment
            if left <= 1e-9:
                break
        left = moved
        for plan in donors:
            capacity = float(plan["target_duration_seconds"]) - _duration_bounds(plan)[0]
            decrement = min(left, capacity)
            plan["target_duration_seconds"] = float(plan["target_duration_seconds"]) - decrement
            left -= decrement
            if left <= 1e-9:
                break
        delta += -moved if delta > 0 else moved
        if abs(delta) <= 1e-6:
            break
    if abs(delta) > 1e-5:
        raise ValueError("plan mix cannot reach the video-task duration quota")
    for plan in plans:
        plan["target_duration_seconds"] = round(float(plan["target_duration_seconds"]), 6)


def _select_video_plans(
    plans: list[dict[str, Any]], video_target: float, group_target: float
) -> tuple[set[str], float]:
    by_class = {
        duration_class: sorted(
            (plan for plan in plans if plan["duration_class"] == duration_class),
            key=lambda plan: stable_hash(plan["plan_id"]),
        )
        for duration_class in ("short", "medium", "long")
    }
    desired_count = max(1, round(len(plans) / 11), int(-(-video_target // 60)))
    total_lower = sum(_duration_bounds(plan)[0] for plan in plans)
    total_upper = sum(_duration_bounds(plan)[1] for plan in plans)
    best: tuple[tuple[float, int, float], tuple[int, int, int], float] | None = None
    candidate_counts = sorted(range(1, len(plans)), key=lambda count: abs(count - desired_count))
    for count in candidate_counts:
        minimum_short = max(0, count - len(by_class["medium"]) - len(by_class["long"]))
        maximum_short = min(count, len(by_class["short"]))
        for short_count in range(minimum_short, maximum_short + 1):
            medium_count = count - short_count
            minimum_medium = max(0, medium_count - len(by_class["long"]))
            maximum_medium = min(medium_count, len(by_class["medium"]))
            for selected_medium in range(minimum_medium, maximum_medium + 1):
                long_count = count - short_count - selected_medium
                lower = 4 * short_count + 16 * selected_medium + 32 * long_count
                upper = 16 * short_count + 32 * selected_medium + 60 * long_count
                dialogue_lower = total_lower - lower
                dialogue_upper = total_upper - upper
                feasible_lower = max(lower, group_target - dialogue_upper)
                feasible_upper = min(upper, group_target - dialogue_lower)
                if feasible_lower > feasible_upper:
                    continue
                attainable = min(max(video_target, feasible_lower), feasible_upper)
                score = (
                    abs(attainable - video_target),
                    abs(count - desired_count),
                    abs((feasible_lower + feasible_upper) / 2 - video_target),
                )
                if best is None or score < best[0]:
                    best = (
                        score,
                        (short_count, selected_medium, long_count),
                        attainable,
                    )
        if best is not None and best[0][0] <= 1e-9 and best[0][1] == 0:
            break
    if best is None:
        raise ValueError("no duration-class assignment can satisfy the video quota")
    counts = best[1]
    selected = {
        plan["plan_id"]
        for duration_class, count in zip(("short", "medium", "long"), counts, strict=True)
        for plan in by_class[duration_class][:count]
    }
    return selected, best[2]


def _calibrate_plan_mix(plans: list[dict[str, Any]], dataset: str) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for plan in plans:
        groups.setdefault((str(plan["language"]), str(plan["split"])), []).append(plan)
    total_seconds = dataset_spec(dataset).duration_seconds
    for (language, split), group in groups.items():
        requested_video = (
            total_seconds * 0.05 * LANGUAGE_FRACTIONS[language] * SPLIT_FRACTIONS[split]
        )
        group_target = total_seconds * 0.55 * LANGUAGE_FRACTIONS[language] * SPLIT_FRACTIONS[split]
        video_ids, attainable_video = _select_video_plans(group, requested_video, group_target)
        for plan in group:
            plan["category"] = (
                "video_task" if plan["plan_id"] in video_ids else "synthetic_dialogue"
            )
        _allocate_duration(group, group_target)
        _shift_video_duration(group, attainable_video)
    _shift_video_duration(plans, total_seconds * 0.05)
    for plan in plans:
        plan["recipe_sha256"] = plan_recipe_sha256(plan)


def _plan(
    dataset: str,
    index: int,
    duration_class: str,
    language: str,
    intent: str,
    split: str,
    fixture: bool,
    plan_offset: int,
) -> dict[str, Any]:
    copy = (ZH_COPY if language == "zh" else EN_COPY)[intent]
    multi = index % 20 >= 12
    turns: list[dict[str, str]] = [
        {"turn_id": "turn-00", "role": "user", "text": copy[0]},
        {"turn_id": "turn-01", "role": "assistant", "text": copy[1]},
    ]
    if multi:
        turns.extend(
            [
                {
                    "turn_id": "turn-02",
                    "role": "user",
                    "text": "继续。" if language == "zh" else "Continue.",
                },
                {"turn_id": "turn-03", "role": "assistant", "text": copy[2]},
                {
                    "turn_id": "turn-04",
                    "role": "user",
                    "text": "好的。" if language == "zh" else "Okay.",
                },
                {"turn_id": "turn-05", "role": "assistant", "text": copy[2]},
            ]
        )
    duration_seconds = {
        "short": 4 + index % 9,
        "medium": 16 + index % 9,
        "long": 32 + index % 9,
    }[duration_class]
    category = "video_task" if fixture and index in {0, 9, 10} else "synthetic_dialogue"
    # IDs are stable within the Canary data catalog.
    plan_id = f"plan-{language}-{plan_offset + index:04d}"
    template_id = f"{dataset}-{intent}-{split}-v{index % 7}"
    scenario_id = f"{dataset}-{split}-{intent}-{index:04d}"
    value = {
        "plan_id": plan_id,
        "dataset": dataset,
        "category": category,
        "language": language,
        "split": split,
        "intent": intent,
        "template_id": template_id,
        "scenario_id": scenario_id,
        "duration_class": duration_class,
        "target_duration_seconds": duration_seconds,
        "turns": turns,
        # Text plans are generated deterministically and admitted by machine
        # gates downstream; there is no manual approval state in this pipeline.
        "quality": {
            "status": "generated",
            "generator": "latentloop-canary-text-v1",
            "fixture": fixture,
        },
    }
    value["recipe_sha256"] = plan_recipe_sha256(value)
    return value


def build_canary_text(
    root: str | Path,
    *,
    fixture: bool = False,
    seed: int = 17,
) -> dict[str, Any]:
    dataset = "canary"
    root = Path(root).expanduser().resolve()
    ensure_tree(root)
    if fixture:
        language_counts = {"zh": 12, "en": 12}
    else:
        language_counts = {"zh": 96, "en": 24}
    plans: list[dict[str, Any]] = []
    duration_offsets: dict[tuple[str, str], int] = {}
    if fixture:
        offsets = {"canary": {"zh": 0, "en": 0}}
    else:
        offsets = {"canary": {"zh": 0, "en": 0}}
    for language, count in language_counts.items():
        intents = _weighted_intents(count)
        random.Random(seed + (0 if language == "zh" else 1)).shuffle(intents)
        for index in range(count):
            split = _split(index, count)
            duration_offset = duration_offsets.get((language, split), 0)
            if fixture:
                duration_class = "short"
            else:
                short_count, medium_count, _ = _FORMAL_DURATION_COUNTS[(language, split)]
                duration_class = (
                    "short"
                    if duration_offset < short_count
                    else "medium"
                    if duration_offset < short_count + medium_count
                    else "long"
                )
            duration_offsets[(language, split)] = duration_offset + 1
            plans.append(
                _plan(
                    dataset,
                    index,
                    duration_class,
                    language,
                    intents[index],
                    split,
                    fixture,
                    offsets[dataset][language],
                )
            )
    if not fixture:
        _calibrate_plan_mix(plans, dataset)
    path = dataset_path(root, dataset, "text", "plans.json")
    value = {
        "dataset": dataset,
        "fixture": fixture,
        "seed": seed,
        "plans": plans,
    }
    write_json(path, value)
    assistant_responses = sum(
        turn["role"] == "assistant" for plan in plans for turn in plan["turns"]
    )
    report = {
        "path": str(path),
        "sha256": sha256_file(path),
        "plans": len(plans),
        "assistant_responses": assistant_responses,
        "languages": language_counts,
        "quality_status": "generated",
    }
    write_json(dataset_path(root, dataset, "reports", "text.json"), report)
    return report
