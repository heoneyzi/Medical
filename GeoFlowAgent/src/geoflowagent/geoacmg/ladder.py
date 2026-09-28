"""RQ2: does generating a whole plan beat choosing one tool at a time?

Every rung of the ladder runs the same episode loop against the same contract
engine, the same corpus and the same cost accounting.  What differs is one thing
only: where the next action comes from.

A planner is a :class:`PlanSource` -- given the current state it proposes an
ordered plan.  Greedy proposes a plan of length one.  A flow model proposes a
whole plan.  The commit length ``k`` decides how much of the proposal is executed
before the world is consulted again, which turns the "plan or react" question
into a curve instead of a coin toss.

Deliberately no torch in this module.  A trained flow model plugs in by
implementing ``propose``; that keeps the ladder testable without a GPU and keeps
the comparison honest, because every rung goes through identical machinery.
"""

from __future__ import annotations

import collections
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

STOP = "<STOP>"


class PlanSource(Protocol):
    """Proposes an ordered list of tool ids from the current state."""

    name: str

    def propose(
        self, state: Mapping[str, Any], candidates: Sequence[str], context: Mapping[str, Any]
    ) -> list[str]:
        ...


@dataclass
class Budget:
    """Three currencies, reported separately because they are not interchangeable.

    In this domain the tool call dominates: an API query costs seconds, while a
    forward pass costs milliseconds.  ``tool_cost`` is therefore the axis every
    comparison is cut on, and the other two are reported so a reviewer can see
    what a rung spent to get there.
    """

    tool_cost: float = 0.0
    calls: int = 0
    plan_evaluations: int = 0
    """Forward passes / ODE integrations spent proposing plans."""

    def add_call(self, cost: float) -> None:
        self.tool_cost += float(cost)
        self.calls += 1

    def to_dict(self) -> dict[str, float]:
        return {
            "tool_cost": self.tool_cost,
            "calls": self.calls,
            "plan_evaluations": self.plan_evaluations,
        }


@dataclass
class Episode:
    task_id: str
    gene: str
    actions: list[str] = field(default_factory=list)
    budget: Budget = field(default_factory=Budget)
    solved: bool = False
    terminated: bool = False
    replans: int = 0
    points_trace: list[int] = field(default_factory=list)
    """Cumulative ACMG points after each action: the step-wise progress signal.

    Cumulative rather than per-action, so the last entry is the evidence the
    episode actually assembled and ``progress_efficiency`` divides a total by a
    total.
    """

    @property
    def progress_efficiency(self) -> float:
        """Points of evidence gathered per unit of tool cost.

        The per-step quantity the benchmark exists to measure.  Zero cost means
        no calls were made, which is reported as zero rather than infinity.
        """

        if self.budget.tool_cost <= 0:
            return 0.0
        return (self.points_trace[-1] if self.points_trace else 0) / self.budget.tool_cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "gene": self.gene,
            "actions": list(self.actions),
            "solved": self.solved,
            "terminated": self.terminated,
            "replans": self.replans,
            "points_trace": list(self.points_trace),
            "progress_efficiency": self.progress_efficiency,
            **self.budget.to_dict(),
        }


# --------------------------------------------------------------------- rungs


@dataclass
class Greedy:
    """One tool at a time from a scorer. The "just split the tools up" baseline."""

    scorer: Callable[[Mapping[str, Any], Sequence[str], Mapping[str, Any]], Mapping[str, float]]
    name: str = "greedy"

    def propose(self, state, candidates, context):
        scores = self.scorer(state, candidates, context)
        return [min(candidates, key=lambda tool: scores.get(tool, math.inf))]


@dataclass
class Beam:
    """Greedy with lookahead, at a plan-evaluation cost that is recorded.

    Included because a flow planner that only wins by effectively searching
    should be matched by a search given the same budget.
    """

    scorer: Callable[[Mapping[str, Any], Sequence[str], Mapping[str, Any]], Mapping[str, float]]
    width: int = 4
    name: str = "beam"

    def propose(self, state, candidates, context):
        scores = self.scorer(state, candidates, context)
        ordered = sorted(candidates, key=lambda tool: scores.get(tool, math.inf))
        return ordered[: self.width]


@dataclass
class WholePlan:
    """A plan source that emits an entire ordering in one shot.

    This is the shape a flow or autoregressive planner has.  ``sampler`` returns
    an ordered tool list; wrapping it here means the flow model never has to know
    about episodes, commit lengths or budgets.
    """

    sampler: Callable[[Mapping[str, Any], Sequence[str], Mapping[str, Any]], list[str]]
    name: str = "whole_plan"
    evaluations_per_call: int = 16
    """NFE for one proposal; recorded in the budget, not charged to tool cost."""

    def propose(self, state, candidates, context):
        plan = [tool for tool in self.sampler(state, candidates, context) if tool in candidates]
        seen: set[str] = set()
        unique = [t for t in plan if not (t in seen or seen.add(t))]
        return unique or list(candidates[:1])


@dataclass
class RandomPolicy:
    seed: int = 17
    name: str = "random"

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def propose(self, state, candidates, context):
        return [self._rng.choice(list(candidates))]


# ------------------------------------------------------------------- runner


def run_episode(
    *,
    planner: PlanSource,
    task_id: str,
    gene: str,
    initial_state: Mapping[str, Any],
    candidates_fn: Callable[[Mapping[str, Any]], list[str]],
    execute_fn: Callable[[Mapping[str, Any], str], tuple[Mapping[str, Any], float, int]],
    solved_fn: Callable[[Mapping[str, Any]], bool],
    terminal_fn: Callable[[Mapping[str, Any]], bool],
    commit: int = 1,
    max_steps: int = 12,
    replan_trigger: Callable[[Mapping[str, Any], str, int, int], bool] | None = None,
    observe: bool = True,
) -> Episode:
    """Run one task under one planner.

    ``commit`` is how many actions of a proposal are executed before replanning.
    ``commit=1`` is eager replanning; ``commit=max_steps`` is open loop.  The
    literature reports that replanning at every step can be the *worst* setting,
    so this is swept rather than assumed.

    ``observe=False`` is the compute-matched blind control: the planner is
    re-invoked the same number of times, with the same budget, but is shown the
    state it started from rather than what the tools returned.  Without it, any
    gain from replanning could just be a gain from more forward passes.

    A planner that ignores the state is unaffected by this control by
    construction, so the hindsight rung shows no blind/observed gap.  That is a
    sanity property worth checking rather than a defect: if it ever *does* show a
    gap, the harness is leaking state into a planner that should not have it.
    """

    episode = Episode(task_id=task_id, gene=gene)
    state: Mapping[str, Any] = dict(initial_state)
    frozen: Mapping[str, Any] = dict(initial_state)
    steps = 0
    while steps < max_steps and not terminal_fn(state):
        candidates = candidates_fn(state)
        if not candidates:
            break
        plan = planner.propose(state if observe else frozen, candidates, {"step": steps})
        episode.budget.plan_evaluations += getattr(planner, "evaluations_per_call", 1)
        episode.replans += 1
        executed = 0
        for tool in plan[: max(1, commit)]:
            if tool == STOP or terminal_fn(state):
                break
            if tool not in candidates_fn(state):
                break  # the world moved; the rest of the plan is stale
            state, cost, points = execute_fn(state, tool)
            episode.budget.add_call(cost)
            episode.actions.append(tool)
            # Evidence *gathered*, so magnitude rather than sign: BA1 at -8 and
            # PVS1 at +8 are both eight points of evidence, and a benign variant
            # must not score as negative progress.
            running = (episode.points_trace[-1] if episode.points_trace else 0) + abs(points)
            episode.points_trace.append(running)
            steps += 1
            executed += 1
            if replan_trigger is not None and replan_trigger(state, tool, steps, executed):
                break
            if steps >= max_steps:
                break
        if executed == 0:
            break
    episode.terminated = terminal_fn(state)
    episode.solved = solved_fn(state)
    return episode


def commit_curve(
    episodes_by_commit: Mapping[int, Sequence[Episode]],
) -> dict[str, Any]:
    """Success and progress efficiency as a function of commit length.

    The shape of this curve is the answer to RQ2.  A minimum at ``k=1`` says the
    whole plan adds nothing beyond its first action; a minimum at ``k>1`` says the
    generated plan buys real lookahead.  Either is a result.
    """

    rows = []
    for commit in sorted(episodes_by_commit):
        episodes = episodes_by_commit[commit]
        if not episodes:
            continue
        rows.append(
            {
                "commit": commit,
                "episodes": len(episodes),
                "success_rate": sum(e.solved for e in episodes) / len(episodes),
                "mean_tool_cost": sum(e.budget.tool_cost for e in episodes) / len(episodes),
                "mean_calls": sum(e.budget.calls for e in episodes) / len(episodes),
                "mean_plan_evaluations": (
                    sum(e.budget.plan_evaluations for e in episodes) / len(episodes)
                ),
                "mean_progress_efficiency": (
                    sum(e.progress_efficiency for e in episodes) / len(episodes)
                ),
                "replans": sum(e.replans for e in episodes) / len(episodes),
            }
        )
    best = max(rows, key=lambda row: row["success_rate"]) if rows else None
    return {
        "curve": rows,
        "best_commit": best["commit"] if best else None,
        "reading": (
            "best_commit == 1 means whole-plan generation adds nothing beyond its first "
            "action in this domain; best_commit > 1 means the plan buys lookahead. Read "
            "at equal mean_tool_cost, not at equal step count"
        ),
    }


def budget_matched(
    episodes: Sequence[Episode], ceiling: float
) -> list[Episode]:
    """Keep only what fits under a tool-cost ceiling.

    Cutting every rung at the same ceiling is what makes the comparison a
    comparison; without it a planner can buy accuracy with calls.
    """

    return [episode for episode in episodes if episode.budget.tool_cost <= ceiling]


def summarise(episodes: Sequence[Episode]) -> dict[str, Any]:
    by_gene: dict[str, list[Episode]] = collections.defaultdict(list)
    for episode in episodes:
        by_gene[episode.gene].append(episode)
    return {
        "episodes": len(episodes),
        "genes": len(by_gene),
        "success_rate": (
            sum(e.solved for e in episodes) / len(episodes) if episodes else 0.0
        ),
        "mean_tool_cost": (
            sum(e.budget.tool_cost for e in episodes) / len(episodes) if episodes else 0.0
        ),
        "mean_progress_efficiency": (
            sum(e.progress_efficiency for e in episodes) / len(episodes) if episodes else 0.0
        ),
        "unterminated": sum(1 for e in episodes if not e.terminated),
    }
