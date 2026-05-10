from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import torch

# Allow running as `python tests/test_core.py` from repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rase.bc import BCAgent, BCConfig
from rase.candidate import SweepConfig, run_candidate_sweep
from rase.fqe import FQEAgent, FQEConfig
from rase.iql import IQLAgent, IQLConfig
from rase.metrics import auprc_from_scores, auroc_from_scores, precision_at_coverage
from rase.networks import GaussianPolicy
from rase.proxy import ProxyConfig, add_knn_proxy_to_selected, proxy_alignment_summary
from rase.replay import D4RLReplayBuffer
from rase.selection import CandidateSelectionConfig, collect_selected_candidates
from rase.utils import set_seed


def make_replay(n=256, obs_dim=5, act_dim=2):
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(n, obs_dim)).astype(np.float32)
    actions = np.tanh(rng.normal(size=(n, act_dim))).astype(np.float32)
    next_obs = (obs + 0.1 * rng.normal(size=(n, obs_dim))).astype(np.float32)
    rewards = rng.normal(size=(n,)).astype(np.float32)
    terminals = np.zeros((n,), dtype=np.float32)
    return D4RLReplayBuffer({
        "observations": obs,
        "actions": actions,
        "next_observations": next_obs,
        "rewards": rewards,
        "terminals": terminals,
    }, normalize_obs=True)


def make_agents(replay, device):
    iql = IQLAgent(replay.obs_dim, replay.act_dim, IQLConfig(hidden_dim=16), device)
    bc = BCAgent(replay.obs_dim, replay.act_dim, BCConfig(hidden_dim=16), device)
    fqe = FQEAgent(replay.obs_dim, replay.act_dim, FQEConfig(hidden_dim=16), ref_policy=iql.actor, device=device)
    return iql, bc, fqe


def test_policy_log_prob_and_sample_n():
    set_seed(0)
    policy = GaussianPolicy(5, 2)
    obs = torch.randn(7, 5)
    acts = policy.sample_n(obs, 4)
    assert acts.shape == (7, 4, 2)
    assert torch.isfinite(acts).all()
    assert acts.min() >= -1.0 and acts.max() <= 1.0
    lp = policy.log_prob(obs, acts[:, 0, :])
    assert lp.shape == (7, 1)
    assert torch.isfinite(lp).all()


def test_candidate_sweep_runs_without_d4rl():
    set_seed(1)
    device = torch.device("cpu")
    replay = make_replay()
    iql, bc, fqe = make_agents(replay, device)
    cfg = SweepConfig(candidate_ms=[1, 4, 8], n_eval_states=32, batch_size=16, source="bc")
    out = run_candidate_sweep(replay, iql, bc, fqe, cfg, device)
    assert len(out["sweep"]) == 3
    assert len(out["risk_coverage"]) == 3 * len(list(cfg.thresholds))
    for row in out["sweep"]:
        assert row["n"] == 32
        for key in ["pred_adv_mean", "fqe_adv_mean", "fpi_rate_cond_pred_positive"]:
            assert math.isfinite(float(row[key]))


def test_selection_and_proxy_alignment_run_without_d4rl():
    set_seed(2)
    device = torch.device("cpu")
    replay = make_replay(n=128)
    iql, bc, fqe = make_agents(replay, device)
    cfg = CandidateSelectionConfig(candidate_ms=[1, 4], n_eval_states=24, batch_size=12, source="bc")
    selected = collect_selected_candidates(replay, iql, bc, fqe, cfg, device)
    assert set(selected.keys()) == {1, 4}
    for M, data in selected.items():
        assert data["selected_action"].shape == (24, replay.act_dim)
        assert data["pred_pair_gap"].shape == (24,)
    proxy_cfg = ProxyConfig(knn_ref_size=32, knn_batch_size=16, coverages=(0.2, 0.5))
    selected = add_knn_proxy_to_selected(selected, replay, proxy_cfg, device)
    rows = proxy_alignment_summary(selected, "bc", proxy_cfg)
    assert rows
    assert all("auroc_fqe_positive" in r for r in rows)


def test_metric_helpers():
    labels = np.array([0, 0, 1, 1], dtype=bool)
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert abs(auroc_from_scores(labels, scores) - 1.0) < 1e-8
    assert abs(auprc_from_scores(labels, scores) - 1.0) < 1e-8
    assert abs(precision_at_coverage(labels, scores, 0.5) - 1.0) < 1e-8


if __name__ == "__main__":
    test_policy_log_prob_and_sample_n()
    test_candidate_sweep_runs_without_d4rl()
    test_selection_and_proxy_alignment_run_without_d4rl()
    test_metric_helpers()
    print("tests_ok")
