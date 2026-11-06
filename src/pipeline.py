import random

from src.kv_model import KVLifecycleModel
from src.latency_oracle import LatencyOracle
from src.model_loader import load_model_config
from src.mapping_overhead import mapping_computation_charge
from src.trace_loader import load_trace_jsonl


class HyperSimulator:

    def __init__(self,
                 policy: str,
                 tech1: bool = False,
                 tech2: bool = False,
                 tech3: bool = False,
                 orches: bool = False,
                 fast_pim_model: bool = False,
                 controller_cost_scale: float = 1.0,
                 beta_threshold: float = 0.2,
                 orches_boundary: str = "action",
                 orches_ab: bool = False,
                 gpu_cp_tflops: float | None = None,
                 pim_bw_scale: float = 1.0,
                 macs_per_bank_scale: float = 1.0,
                 lambda_mig: float = 0.15,
                 spec_hit_rate: float = 0.8,
                 charge_mapping_compute: bool = True):
        if not 0.0 <= beta_threshold <= 1.0:
            raise ValueError("beta_threshold must be in [0, 1]")
        if orches_boundary not in ("action", "branch"):
            raise ValueError("orches_boundary must be 'action' or 'branch'")
        if pim_bw_scale <= 0 or macs_per_bank_scale <= 0:
            raise ValueError("PIM scales must be positive")
        self.policy = policy
        self.orches = orches if policy == "ours" else False
        self.tech1 = tech1 if policy == "ours" and not self.orches else False
        self.tech2 = tech2 if policy == "ours" and not self.orches else False
        self.tech3 = tech3 if policy == "ours" and not self.orches else False
        self.fast_pim_model = fast_pim_model
        self.controller_cost_scale = controller_cost_scale
        self.beta_threshold = beta_threshold
        self.orches_boundary = orches_boundary
        self.orches_ab = orches_ab
        self.gpu_cp_tflops = gpu_cp_tflops
        self.pim_bw_scale = pim_bw_scale
        self.macs_per_bank_scale = macs_per_bank_scale
        self.lambda_mig = lambda_mig
        self.hw_bw_link = 204.8e9
        self.hw_bw_pim_int = 24.2e12 * pim_bw_scale
        self.spec_hit_rate = spec_hit_rate
        self.charge_mapping_compute = charge_mapping_compute

    def _baseline_control_path(self, step: dict, control_bytes: float) -> tuple:
        controller_ms = step.get("controller", {}).get("controller_time_ms", 0.0) * self.controller_cost_scale
        if self.policy in ["baseline_gpu", "baseline_gpu_sw"]:
            return controller_ms, 0.0
        handshake_bytes = control_bytes + 8 * 1024
        if self.policy == "baseline_static":
            handshake_bytes *= 1.4
            alive = step.get("alive_paths", 0)
            bubble_ms = (handshake_bytes / self.hw_bw_link) * 1000.0 + self.controller_cost_scale * (2.8 + 0.012 * alive)
        else:
            bubble_ms = (handshake_bytes / self.hw_bw_link) * 1000.0 + self.controller_cost_scale * 0.012
        return controller_ms, bubble_ms

    def _decide_and_bubble(self, step: dict, control_bytes: float) -> tuple:
        if not self.tech3:
            decide_ms, bubble_ms = self._baseline_control_path(step, control_bytes)
            return decide_ms, bubble_ms, {
                "spec_hit": None,
                "commit_ms": 0.0,
                "rollback_ms": 0.0,
                "slowpath_ms": 0.0,
            }

        decision_id = int(step.get("decision_id", 0))
        rng = random.Random(20260405 + decision_id)
        hit = rng.random() < self.spec_hit_rate
        control_sync_ms = (control_bytes / self.hw_bw_link) * 1000.0
        commit_ms = self.controller_cost_scale * (0.0008 + 0.05 * control_sync_ms)
        rollback_ms = self.controller_cost_scale * (0.00025 + 0.02 * control_sync_ms)
        slow_decide_ms, slow_bubble_ms = self._baseline_control_path(step, control_bytes)
        slowpath_ms = slow_decide_ms + slow_bubble_ms
        total_sync_ms = commit_ms if hit else (rollback_ms + slowpath_ms)
        return total_sync_ms, 0.0, {
            "spec_hit": hit,
            "commit_ms": commit_ms if hit else 0.0,
            "rollback_ms": 0.0 if hit else rollback_ms,
            "slowpath_ms": 0.0 if hit else slowpath_ms,
        }

    def run(self, trace_path: str, model_config_path: str) -> dict:
        trace = load_trace_jsonl(trace_path)
        model_cfg = load_model_config(model_config_path)
        oracle = LatencyOracle(model_cfg,
                               fast_pim_model=self.fast_pim_model,
                               orches=self.orches)
        if self.gpu_cp_tflops is not None:
            oracle.hw.cp_gpu = self.gpu_cp_tflops * 1e12
        oracle.hw.bw_pim_int = self.hw_bw_pim_int
        oracle.hw.cp_pim *= self.macs_per_bank_scale
        oracle.lambda_mig = self.lambda_mig
        kv_model = KVLifecycleModel(tech2=self.tech2,
                                    orches=self.orches,
                                    orches_ab=self.orches_ab,
                                    bw_link=self.hw_bw_link,
                                    bw_pim_int=self.hw_bw_pim_int,
                                    beta_threshold=self.beta_threshold,
                                    controller_cost_scale=self.controller_cost_scale)

        totals = {
            "total_latency_ms": 0.0,
            "gpu_compute_ms": 0.0,
            "pim_compute_ms": 0.0,
            "pcie_comm_ms": 0.0,
            "kv_mgmt_ms": 0.0,
            "sync_bubble_ms": 0.0,
            "tech3_commit_ms": 0.0,
            "tech3_rollback_ms": 0.0,
            "tech3_slowpath_ms": 0.0,
            "tech3_hit_count": 0,
            "tech3_total_count": 0,
            "mapping_compute_ms": 0.0,
            "mapping_attention_ms": 0.0,
            "mapping_moe_ms": 0.0,
            "mapping_invocations": 0,
        }
        step_breakdown = []
        alpha_prev = 0.0
        gamma_prev = 0.0

        for step_index, step in enumerate(trace):
            # Reconstruct ORCHES-A's branch-round coordination. ORCHES-AB
            # retains the same mapping and software KV path but may remap at
            # every action boundary. The legacy default remains action-level.
            normal_alpha_candidates = oracle.alpha_candidates
            if (self.orches and self.orches_boundary == "branch"
                    and step_index > 0
                    and step.get("action") not in ("branch", "multitok")):
                oracle.alpha_candidates = [alpha_prev]
            est = oracle.estimate_epoch(step=step,
                                        policy=self.policy,
                                        tech1=self.tech1,
                                        alpha_prev=alpha_prev,
                                        gamma_prev=gamma_prev)
            oracle.alpha_candidates = normal_alpha_candidates
            kv = kv_model.handle_epoch(step, policy=self.policy)
            decide_ms, bubble_ms, spec = self._decide_and_bubble(step, est["control_bytes"])
            mapper = mapping_computation_charge(
                int(est["moe_activated_experts"]),len(oracle.alpha_candidates),
                invoked=(self.charge_mapping_compute and self.tech1
                         and int(step.get("alive_paths", 0)) > 0))
            generate_ms = est.get("generate_ms", max(est["gpu_compute_ms"], est["pim_compute_ms"]))
            merge_ms = est["merge_ms"]
            pcie_ms = est["pcie_comm_ms"] + (kv["pcie_bytes"] / self.hw_bw_link) * 1000.0
            if self.policy == "ours":
                if self.tech2:
                    pcie_ms *= 0.8
                if self.tech3:
                    pcie_ms *= 0.9
            if self.policy == "baseline_static":
                comm_exposed_ms = pcie_ms * 1.8
            elif self.policy in ["baseline_gpu", "baseline_gpu_sw"]:
                comm_exposed_ms = 0.0
            else:
                comm_exposed_ms = pcie_ms * (0.35 if not self.tech3 else 0.08)
            total_step_ms = (generate_ms + decide_ms + mapper["mapping_ms"]
                             + kv["kv_mgmt_ms"] + bubble_ms + merge_ms
                             + comm_exposed_ms)

            totals["total_latency_ms"] += total_step_ms
            totals["gpu_compute_ms"] += est["gpu_compute_ms"]
            totals["pim_compute_ms"] += est["pim_compute_ms"]
            totals["pcie_comm_ms"] += pcie_ms
            totals["kv_mgmt_ms"] += kv["kv_mgmt_ms"]
            totals["sync_bubble_ms"] += bubble_ms + comm_exposed_ms
            totals["tech3_commit_ms"] += spec["commit_ms"]
            totals["tech3_rollback_ms"] += spec["rollback_ms"]
            totals["tech3_slowpath_ms"] += spec["slowpath_ms"]
            totals["mapping_compute_ms"] += mapper["mapping_ms"]
            totals["mapping_attention_ms"] += mapper["attention_ms"]
            totals["mapping_moe_ms"] += mapper["moe_ms"]
            totals["mapping_invocations"] += int(mapper["mapper_invoked"])
            if spec["spec_hit"] is not None:
                totals["tech3_total_count"] += 1
                totals["tech3_hit_count"] += int(bool(spec["spec_hit"]))

            step_breakdown.append({
                "decision_id": step.get("decision_id"),
                "action": step.get("action"),
                "remap_allowed": (not self.orches or self.orches_boundary == "action"
                                  or step_index == 0
                                  or step.get("action") in ("branch", "multitok")),
                "alpha": est["alpha"],
                "alpha_shared": est.get("alpha_shared", est["alpha"]),
                "gamma": est["gamma"],
                "t_gpu_attn_shared_stage_ms": est.get("t_gpu_attn_shared_stage_ms", 0.0),
                "t_pim_attn_shared_stage_ms": est.get("t_pim_attn_shared_stage_ms", 0.0),
                "attn_residual_gap_ms": est.get("attn_residual_gap_ms", 0.0),
                "unique_rebalance_triggered": est.get("unique_rebalance_triggered", False),
                "gpu_unique_paths": est.get("gpu_unique_paths", 0),
                "gpu_unique_path_ids": est.get("gpu_unique_path_ids", []),
                "t_gpu_attn_final_ms": est.get("t_gpu_attn_final_ms", 0.0),
                "t_pim_attn_final_ms": est.get("t_pim_attn_final_ms", 0.0),
                "tech1_total_makespan_est_ms": est.get("tech1_total_makespan_est_ms", 0.0),
                "route_gpu_shared_attn_ms": est.get("route_gpu_shared_attn_ms", 0.0),
                "route_pim_shared_attn_ms": est.get("route_pim_shared_attn_ms", 0.0),
                "route_gpu_moe_ms": est.get("route_gpu_moe_ms", 0.0),
                "route_pim_moe_ms": est.get("route_pim_moe_ms", 0.0),
                "moe_split_fraction": est.get("moe_split_fraction", 0.0),
                "moe_split_index": est.get("moe_split_index", 0),
                "moe_activated_experts": est.get("moe_activated_experts", 0),
                "moe_fanin_threshold": est.get("moe_fanin_threshold", 0),
                "generate_ms": generate_ms,
                "decide_ms": decide_ms,
                "mapping_compute_ms": mapper["mapping_ms"],
                "mapping_attention_ms": mapper["attention_ms"],
                "mapping_moe_ms": mapper["moe_ms"],
                "mapping_invoked": mapper["mapper_invoked"],
                "mapping_attention_cycles": mapper["attention_cycles"],
                "mapping_moe_cycles": mapper["moe_cycles"],
                "mapping_alpha_candidate_slots": mapper["alpha_candidate_slots"],
                "kv_mgmt_ms": kv["kv_mgmt_ms"],
                "merge_ms": merge_ms,
                "bubble_ms": bubble_ms,
                "spec_hit": spec["spec_hit"],
                "commit_ms": spec["commit_ms"],
                "rollback_ms": spec["rollback_ms"],
                "slowpath_ms": spec["slowpath_ms"],
                "total_step_ms": total_step_ms,
                "gpu_flops": est["gpu_flops"],
                "pim_flops": est["pim_flops"],
                "gpu_bytes": est.get("gpu_bytes", 0.0),
                "pim_bytes": est.get("pim_bytes", 0.0),
                "pcie_bytes": est["pcie_bytes"] + kv["pcie_bytes"],
                "model_pcie_bytes": est["pcie_bytes"],
                "kv_pcie_bytes": kv["pcie_bytes"],
                "control_bytes": est.get("control_bytes", 0.0),
                "merge_bytes": est.get("merge_bytes", 0.0),
                "staged_bytes": est.get("staged_bytes", 0.0),
                "spill_bytes": est.get("spill_bytes", 0.0),
                "kv_moved_blocks": kv["moved_blocks"],
                "kv_compacted_blocks": kv["compacted_blocks"],
                "kv_metadata_overhead_ms": kv["metadata_overhead_ms"],
                "kv_live_blocks": kv["kv_live_blocks"],
                "kv_holes_blocks": kv["kv_holes_blocks"],
                "candidate_seqs": step.get("candidate_seqs", 0),
                "alive_paths": step.get("alive_paths", 0),
                "kv_alloc_blocks_delta": step.get("kv", {}).get("kv_alloc_blocks_delta", 0),
                "kv_free_blocks_delta": step.get("kv", {}).get("kv_free_blocks_delta", 0),
                "beta": kv["beta"],
            })
            alpha_prev = est["alpha"]
            gamma_prev = est["gamma"]

        if totals["tech3_total_count"] > 0:
            totals["tech3_hit_rate"] = totals["tech3_hit_count"] / totals["tech3_total_count"]
        else:
            totals["tech3_hit_rate"] = None

        model_name = model_cfg.get("model_type", "custom")
        return {
            "model": model_name,
            "policy": self.policy,
            "tech": {
                "tech1": self.tech1,
                "tech2": self.tech2,
                "tech3": self.tech3,
                "orches": self.orches
            },
            "metrics": totals,
            "step_breakdown": step_breakdown,
        }
