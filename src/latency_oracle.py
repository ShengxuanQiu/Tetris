import math
import os
from dataclasses import dataclass
import numpy as np
from src.pim_ramulator_runtime import PIMAttentionDescriptor, PIMRamulatorRuntime


@dataclass
class HardwareSpec:
    cp_gpu: float = 43e12
    bw_gpu: float = 204.8e9
    cp_pim: float = 87.2e12
    bw_pim_int: float = 24.2e12
    bw_link: float = 204.8e9
    n_pim_banks: int = 4096
    expert_bank_span: int = 64

    @property
    def cp_bank(self) -> float:
        return self.cp_pim / self.n_pim_banks

    @property
    def bw_bank(self) -> float:
        return self.bw_pim_int / self.n_pim_banks


class LatencyOracle:

    def __init__(self, model_cfg: dict, fast_pim_model: bool = False, orches: bool = False):
        self.model = model_cfg
        self.hw = HardwareSpec()
        self.orches = orches
        self.alpha_candidates = [0.0, 0.25, 0.5, 0.75, 1.0]
        self.lambda_mig = 0.15
        self.fast_pim_model = fast_pim_model
        self.pim_runtime = None if fast_pim_model else PIMRamulatorRuntime()

    @staticmethod
    def _to_ms(seconds: float) -> float:
        return seconds * 1000.0

    def _shared_unique_len(self, step: dict) -> tuple:
        action = step.get("action", "none")
        ctx = max(step.get("ctx", {}).get("ctx_len_mean", 0), 0)
        if action in ["multitok", "branch"]:
            uniq = step.get("tokens_generated_per_seq", 0)
            return max(ctx, 0), max(uniq, 0)
        return 0, max(ctx, 0)

    def _base_workload(self, step: dict) -> dict:
        action = step.get("action", "none")
        ctx_len = max(step.get("ctx", {}).get("ctx_len_mean", 0), 0)
        cand = max(step.get("candidate_seqs", 0), 0)
        tok = max(step.get("tokens_generated_per_seq", 0), 0)
        expand_k = max(step.get("expand_k", 1), 1)
        layers = self.model["num_hidden_layers"]
        h = self.model["hidden_size"]
        kv_heads = self.model["num_key_value_heads"]
        d_head = self.model["head_dim"]
        moe_hidden = self.model["moe_intermediate_size"]
        k_top = self.model["num_experts_per_tok"]

        n_token_ops = cand * tok
        attn_bytes_per_layer = 2 * kv_heads * d_head * 2 * ctx_len
        attn_flops_per_layer = 2 * h * ctx_len
        moe_flops_per_layer = 2 * h * moe_hidden * k_top * 2

        attn_bytes = attn_bytes_per_layer * layers * n_token_ops
        attn_flops = attn_flops_per_layer * layers * n_token_ops
        moe_flops = moe_flops_per_layer * layers * n_token_ops

        # RoE-style singletok inflates token-local compute without replicating full KV history.
        if action == "singletok":
            moe_flops *= expand_k

        return {
            "attn_bytes": float(attn_bytes),
            "attn_flops": float(attn_flops),
            "moe_flops": float(moe_flops),
            "n_token_ops": float(n_token_ops),
            "ctx_len": float(ctx_len),
        }

    def _merge_bytes(self, step: dict) -> float:
        action = step.get("action", "none")
        if action not in ["multitok", "branch"]:
            return 0.0
        cand = max(step.get("candidate_seqs", 0), 0)
        tok = max(step.get("tokens_generated_per_seq", 0), 0)
        h = self.model["hidden_size"]
        heads = self.model["num_attention_heads"]
        partial_bytes_per_tok = heads * 8 + h * 2
        return float(cand * tok * partial_bytes_per_tok)

    def _control_bytes(self, step: dict) -> float:
        alive = max(step.get("alive_paths", 0), 0)
        cand = max(step.get("candidate_seqs", 0), 0)
        return float((alive * 64) + (cand * 128) + 4096)

    def _approx_expert_fanins(self, step: dict) -> list[int]:
        candidate = max(int(step.get("candidate_seqs", 0)), 1)
        expand_k = max(int(step.get("expand_k", 1)), 1)
        token_span = max(int(step.get("tokens_generated_per_seq", step.get("action_span_tokens", 1))), 1)
        k_top = max(int(self.model.get("num_experts_per_tok", 1)), 1)
        num_experts = max(int(self.model.get("num_experts", self.model.get("num_local_experts", 1) or 1)), 1)
        action = step.get("action", "none")

        effective_tokens = candidate * token_span * (expand_k if action == "singletok" else 1)
        total_assignments = max(effective_tokens * k_top, 1)
        n_active = min(num_experts, max(k_top, int(round(1.8 * math.sqrt(total_assignments)))))
        weights = np.array([1.0 / ((i + 1) ** 1.12) for i in range(n_active)], dtype=np.float64)
        weights /= max(weights.sum(), 1e-12)
        raw = np.floor(total_assignments * weights).astype(np.int32)
        raw = np.maximum(raw, 1)
        diff = int(total_assignments - int(raw.sum()))
        if diff > 0:
            order = np.argsort(-weights)
            for i in range(diff):
                raw[order[i % len(order)]] += 1
        elif diff < 0:
            order = np.argsort(weights)
            need = -diff
            idx = 0
            while need > 0 and idx < len(order) * 4:
                j = order[idx % len(order)]
                if raw[j] > 1:
                    raw[j] -= 1
                    need -= 1
                idx += 1
        fanins = sorted((int(x) for x in raw if x > 0), reverse=True)
        return fanins

    def _expert_weight_bytes(self) -> float:
        h = float(self.model["hidden_size"])
        moe_hidden = float(self.model["moe_intermediate_size"])
        return 3.0 * h * moe_hidden * 2.0

    def _expert_flops_per_token(self) -> float:
        h = float(self.model["hidden_size"])
        moe_hidden = float(self.model["moe_intermediate_size"])
        return 4.0 * h * moe_hidden

    def _gpu_expert_time(self, fanin: int) -> float:
        c = self._expert_flops_per_token()
        s_e = self._expert_weight_bytes()
        # Unified-memory fetch cost is a fixed term per activated expert, but it is
        # amortized across the routed tokens that reuse the same fetched weights.
        fetch_amort = min(max(float(fanin), 1.0), 8.0)
        return max((c * fanin) / self.hw.cp_gpu, s_e / (self.hw.bw_link * fetch_amort))

    def _gpu_sw_efficiency(self, step: dict, action: str, shared_len: int, ctx_len: int) -> tuple[float, float]:
        candidate = max(int(step.get("candidate_seqs", 0)), 1)
        base_prompt_len = max(int(step.get("ctx", {}).get("base_prompt_len", 0)), 0)
        logp = math.log2(max(candidate, 1))
        prompt_share = (base_prompt_len / max(ctx_len, 1)) if ctx_len > 0 else 0.0
        topology_share = (shared_len / max(ctx_len, 1)) if ctx_len > 0 else 0.0

        # GPU-SW improves software-only execution through paged KV, prefix aliasing
        # and cleanup batching. This primarily reduces attention-side memory work and
        # mildly improves MoE batching, without invoking any heterogeneous execution.
        attn_gain = 0.04 * min(logp, 5.0) + 0.20 * topology_share + 0.08 * prompt_share
        if action in ["multitok", "branch"]:
            attn_gain += 0.06
        elif action == "singletok":
            attn_gain += 0.03
        else:
            attn_gain += 0.01 * min(logp, 3.0)

        moe_gain = 0.015 * min(logp, 5.0)
        if action == "singletok":
            moe_gain += 0.03
        elif action in ["multitok", "branch"]:
            moe_gain += 0.01

        attn_factor = max(0.70, 1.0 - attn_gain)
        moe_factor = max(0.88, 1.0 - moe_gain)
        return attn_factor, moe_factor

    def _pim_expert_time(self, fanin: int) -> float:
        c = self._expert_flops_per_token()
        s_e = self._expert_weight_bytes()
        n_bank_active = float(self.hw.expert_bank_span)
        cp_eff = max(n_bank_active * self.hw.cp_bank, 1e-12)
        bw_eff = max(n_bank_active * self.hw.bw_bank, 1e-12)
        return max((c * fanin) / cp_eff, s_e / bw_eff)

    def _moe_all_gpu_time(self, step: dict) -> tuple[float, dict]:
        fanins = self._approx_expert_fanins(step)
        t_gpu = sum(self._gpu_expert_time(n) for n in fanins)
        return t_gpu, {
            "split_fraction": 1.0,
            "split_index": len(fanins),
            "activated_experts": len(fanins),
            "fanin_threshold": fanins[-1] if fanins else 0,
        }

    def _moe_dynamic_split(self, step: dict, fixed_gpu_time: float = 0.0, fixed_pim_time: float = 0.0) -> tuple[float, float, dict]:
        fanins = self._approx_expert_fanins(step)
        gpu_times = [self._gpu_expert_time(n) for n in fanins]
        pim_times = [self._pim_expert_time(n) for n in fanins]
        gpu_prefix = [0.0]
        for t in gpu_times:
            gpu_prefix.append(gpu_prefix[-1] + t)
        pim_suffix = [0.0] * (len(fanins) + 1)
        for i in range(len(fanins) - 1, -1, -1):
            pim_suffix[i] = pim_suffix[i + 1] + pim_times[i]

        best = None
        for k in range(len(fanins) + 1):
            t_gpu = fixed_gpu_time + gpu_prefix[k]
            t_pim = fixed_pim_time + pim_suffix[k]
            makespan = max(t_gpu, t_pim)
            if best is None or makespan < best["makespan"]:
                best = {
                    "k": k,
                    "t_gpu": t_gpu,
                    "t_pim": t_pim,
                    "makespan": makespan,
                }
        gpu_moe_time = max(best["t_gpu"] - fixed_gpu_time, 0.0)
        pim_moe_time = max(best["t_pim"] - fixed_pim_time, 0.0)
        return gpu_moe_time, pim_moe_time, {
            "split_fraction": (best["k"] / len(fanins)) if fanins else 0.0,
            "split_index": best["k"],
            "activated_experts": len(fanins),
            "fanin_threshold": fanins[best["k"] - 1] if fanins and best["k"] > 0 else (fanins[0] if fanins else 0),
        }

    def _estimate_pim_attention_time(self,
                                     step: dict,
                                     action: str,
                                     candidate: int,
                                     ctx_len: int,
                                     shared_len: int,
                                     unique_len: int,
                                     alpha: float,
                                     shared_ratio: float,
                                     pim_attn_bytes: float,
                                     pim_attn_flops: float) -> float:
        if pim_attn_bytes <= 0 and pim_attn_flops <= 0:
            return 0.0
        if not self.fast_pim_model and pim_attn_bytes > 0 and self.pim_runtime is not None:
            return self._estimate_pim_attention_runtime(
                step=step,
                action=action,
                candidate=candidate,
                ctx_len=ctx_len,
                shared_len=shared_len,
                unique_len=unique_len,
                alpha=alpha,
                shared_ratio=shared_ratio,
                pim_bytes=pim_attn_bytes,
            )
        pim_bw = self.hw.bw_pim_int * (1.08 if self.fast_pim_model else 1.0)
        return max(pim_attn_flops / self.hw.cp_pim, pim_attn_bytes / pim_bw)

    def _gpu_attention_time(self, attn_flops: float, attn_bytes: float) -> float:
        if attn_flops <= 0 and attn_bytes <= 0:
            return 0.0
        return max(attn_flops / self.hw.cp_gpu, attn_bytes / self.hw.bw_gpu)

    def _pim_attention_time_analytical(self, attn_flops: float, attn_bytes: float) -> float:
        if attn_flops <= 0 and attn_bytes <= 0:
            return 0.0
        pim_bw = self.hw.bw_pim_int * (1.08 if self.fast_pim_model else 1.0)
        return max(attn_flops / self.hw.cp_pim, attn_bytes / pim_bw)

    def _rebalance_unique_attention_paths(self,
                                          candidate: int,
                                          gpu_shared_time: float,
                                          pim_shared_time: float,
                                          unique_flops: float,
                                          unique_bytes: float,
                                          unique_pim_total_time: float) -> dict:
        path_count = max(int(candidate), 1)
        if (unique_flops <= 0 and unique_bytes <= 0) or unique_pim_total_time <= 0:
            return {
                "triggered": False,
                "moved_paths": 0,
                "moved_path_ids": [],
                "gpu_time_final": gpu_shared_time,
                "pim_time_final": pim_shared_time,
                "gpu_unique_time": 0.0,
                "pim_unique_time": unique_pim_total_time,
            }
        unique_flops_per_path = unique_flops / path_count
        unique_bytes_per_path = unique_bytes / path_count

        def gpu_unique_time_for_paths(moved_paths: int) -> float:
            if moved_paths <= 0:
                return 0.0
            moved_flops = unique_flops_per_path * moved_paths
            moved_bytes = unique_bytes_per_path * moved_paths
            # GPU can batch a handful of tail paths together; model this as a
            # bounded bandwidth amortization factor rather than charging each path
            # its own worst-case raw KV stream.
            bundle_gain = min(max(float(moved_paths), 1.0), 8.0)
            effective_bw = self.hw.bw_gpu * bundle_gain
            return max(moved_flops / self.hw.cp_gpu, moved_bytes / effective_bw)

        def pim_unique_time_for_paths(remaining_paths: int) -> float:
            if remaining_paths <= 0:
                return 0.0
            return unique_pim_total_time * (remaining_paths / path_count)

        gpu_time = gpu_shared_time
        pim_time = pim_shared_time + unique_pim_total_time
        moved = []
        for path_id in range(path_count):
            current_gap = abs(pim_time - gpu_time)
            moved_count = len(moved) + 1
            remaining_count = path_count - moved_count
            new_gpu = gpu_shared_time + gpu_unique_time_for_paths(moved_count)
            new_pim = pim_shared_time + pim_unique_time_for_paths(remaining_count)
            new_gap = abs(new_pim - new_gpu)
            current_obj = max(gpu_time, pim_time)
            new_obj = max(new_gpu, new_pim)
            if new_gap >= current_gap and new_obj > current_obj:
                break
            if new_gpu > new_pim and new_obj >= current_obj and moved:
                break
            gpu_time = new_gpu
            pim_time = new_pim
            moved.append(path_id)

        moved_count = len(moved)
        gpu_unique_time = gpu_unique_time_for_paths(moved_count)
        pim_unique_time = pim_unique_time_for_paths(path_count - moved_count)
        return {
            "triggered": moved_count > 0,
            "moved_paths": moved_count,
            "moved_path_ids": moved,
            "gpu_time_final": gpu_time,
            "pim_time_final": pim_time,
            "gpu_unique_time": gpu_unique_time,
            "pim_unique_time": pim_unique_time,
        }

    def _schedule_attention_two_stage(self,
                                      step: dict,
                                      action: str,
                                      candidate: int,
                                      ctx_len: int,
                                      shared_len: int,
                                      unique_len: int,
                                      shared_flops: float,
                                      shared_bytes: float,
                                      unique_flops: float,
                                      unique_bytes: float,
                                      alpha_prev: float) -> dict:
        alpha_space = self.alpha_candidates if shared_flops > 0 and shared_bytes > 0 else [0.0]
        best = None
        total_attn = shared_flops + unique_flops
        shared_ratio = (shared_flops / max(total_attn, 1e-12)) if total_attn > 0 else 0.0
        for alpha_shared in alpha_space:
            gpu_shared_flops = alpha_shared * shared_flops
            gpu_shared_bytes = alpha_shared * shared_bytes
            pim_shared_flops = (1.0 - alpha_shared) * shared_flops
            pim_shared_bytes = (1.0 - alpha_shared) * shared_bytes

            t_gpu_shared = self._gpu_attention_time(gpu_shared_flops, gpu_shared_bytes)
            t_pim_shared_only = self._estimate_pim_attention_time(
                step=step,
                action=action,
                candidate=candidate,
                ctx_len=ctx_len,
                shared_len=shared_len,
                unique_len=0,
                alpha=alpha_shared,
                shared_ratio=shared_ratio,
                pim_attn_bytes=pim_shared_bytes,
                pim_attn_flops=pim_shared_flops,
            )
            t_pim_shared = self._estimate_pim_attention_time(
                step=step,
                action=action,
                candidate=candidate,
                ctx_len=ctx_len,
                shared_len=shared_len,
                unique_len=unique_len,
                alpha=alpha_shared,
                shared_ratio=shared_ratio,
                pim_attn_bytes=pim_shared_bytes + unique_bytes,
                pim_attn_flops=pim_shared_flops + unique_flops,
            )
            residual_gap = t_pim_shared - t_gpu_shared
            unique_sched = {
                "triggered": False,
                "moved_paths": 0,
                "moved_path_ids": [],
                "gpu_time_final": t_gpu_shared,
                "pim_time_final": t_pim_shared,
            }
            t_gpu_final = t_gpu_shared
            t_pim_final = t_pim_shared
            c_mig = abs(alpha_shared - alpha_prev) * shared_bytes * 0.002 / self.hw.bw_link
            objective = max(t_gpu_final, t_pim_final) + self.lambda_mig * c_mig
            if best is None or objective < best["objective"]:
                best = {
                    "alpha_shared": alpha_shared,
                    "t_gpu_attn_shared_only": t_gpu_shared,
                    "t_pim_attn_shared_only": t_pim_shared_only,
                    "t_gpu_attn_shared_stage": t_gpu_shared,
                    "t_pim_attn_shared_stage": t_pim_shared,
                    "attn_residual_gap": residual_gap,
                    "unique_rebalance_triggered": False,
                    "gpu_unique_paths": unique_sched["moved_paths"],
                    "gpu_unique_path_ids": unique_sched["moved_path_ids"],
                    "t_gpu_attn_final": t_gpu_final,
                    "t_pim_attn_final": t_pim_final,
                    "gpu_attn_flops": gpu_shared_flops,
                    "gpu_attn_bytes": gpu_shared_bytes,
                    "pim_attn_flops": pim_shared_flops + unique_flops,
                    "pim_attn_bytes": pim_shared_bytes + unique_bytes,
                    "objective": objective,
                }
        return best

    def _search_alpha_gamma(self,
                            step: dict,
                            action: str,
                            candidate: int,
                            ctx_len: int,
                            shared_len: int,
                            unique_len: int,
                            shared_ratio: float,
                            shared_flops: float,
                            shared_bytes: float,
                            unique_flops: float,
                            unique_bytes: float,
                            moe_flops: float,
                            merge_bytes: float,
                            alpha_prev: float,
                            gamma_prev: float) -> tuple:
        attn_detail = self._schedule_attention_two_stage(
            step=step,
            action=action,
            candidate=candidate,
            ctx_len=ctx_len,
            shared_len=shared_len,
            unique_len=unique_len,
            shared_flops=shared_flops,
            shared_bytes=shared_bytes,
            unique_flops=unique_flops,
            unique_bytes=unique_bytes,
            alpha_prev=alpha_prev,
        )

        alpha = attn_detail["alpha_shared"]
        gpu_attn_flops = attn_detail["gpu_attn_flops"]
        gpu_attn_bytes = attn_detail["gpu_attn_bytes"]
        pim_attn_flops = attn_detail["pim_attn_flops"]
        pim_attn_bytes = attn_detail["pim_attn_bytes"]
        t_attn_gpu = attn_detail["t_gpu_attn_final"]
        t_attn_pim = attn_detail["t_pim_attn_final"]

        if self.orches:
            t_moe_gpu, split_info = self._moe_all_gpu_time(step)
            t_moe_pim = 0.0
        else:
            t_moe_gpu, t_moe_pim, split_info = self._moe_dynamic_split(
                step,
                fixed_gpu_time=0.0,
                fixed_pim_time=0.0,
            )

        detail = {
            "alpha": alpha,
            "gamma": 0.0,
            "objective": max(t_attn_gpu, t_attn_pim) + max(t_moe_gpu, t_moe_pim),
            "gpu_flops": gpu_attn_flops,
            "gpu_bytes": gpu_attn_bytes,
            "pim_flops": pim_attn_flops,
            "pim_bytes": pim_attn_bytes,
            "spill_bytes": 0.0,
            "gpu_attn_flops": gpu_attn_flops,
            "gpu_attn_bytes": gpu_attn_bytes,
            "gpu_moe_flops": moe_flops,
            "gpu_moe_bytes": 0.0,
            "pim_attn_flops": pim_attn_flops,
            "pim_attn_bytes": pim_attn_bytes,
            "pim_moe_flops": 0.0,
            "pim_moe_bytes": 0.0,
            "t_attn_gpu": t_attn_gpu,
            "t_attn_pim": t_attn_pim,
            "t_moe_gpu": t_moe_gpu,
            "t_moe_pim": t_moe_pim,
            "t_generate": max(t_attn_gpu, t_attn_pim) + max(t_moe_gpu, t_moe_pim),
            "moe_split_fraction": split_info["split_fraction"],
            "moe_split_index": split_info["split_index"],
            "moe_activated_experts": split_info["activated_experts"],
            "moe_fanin_threshold": split_info["fanin_threshold"],
            "alpha_shared": attn_detail["alpha_shared"],
            "t_gpu_attn_shared_stage": attn_detail["t_gpu_attn_shared_stage"],
            "t_pim_attn_shared_stage": attn_detail["t_pim_attn_shared_stage"],
            "attn_residual_gap": attn_detail["attn_residual_gap"],
            "unique_rebalance_triggered": attn_detail["unique_rebalance_triggered"],
            "gpu_unique_paths": attn_detail["gpu_unique_paths"],
            "gpu_unique_path_ids": attn_detail["gpu_unique_path_ids"],
            "t_gpu_attn_final": attn_detail["t_gpu_attn_final"],
            "t_pim_attn_final": attn_detail["t_pim_attn_final"],
            "tech1_total_makespan_est": max(t_attn_gpu, t_attn_pim) + max(t_moe_gpu, t_moe_pim),
        }
        return alpha, 0.0, detail

    def estimate_epoch(self,
                       step: dict,
                       policy: str,
                       tech1: bool,
                       alpha_prev: float,
                       gamma_prev: float) -> dict:
        action = step.get("action", "none")
        wk = self._base_workload(step)
        merge_bytes = self._merge_bytes(step)
        control_bytes = self._control_bytes(step)
        shared_len, unique_len = self._shared_unique_len(step)
        alpha = 0.0
        gamma = 0.0

        attn_flops = wk["attn_flops"]
        attn_bytes = wk["attn_bytes"]
        moe_flops = wk["moe_flops"]
        sum_len = shared_len + unique_len
        shared_ratio = (shared_len / sum_len) if sum_len > 0 else 0.0
        shared_flops = attn_flops * shared_ratio
        unique_flops = attn_flops - shared_flops
        shared_bytes = attn_bytes * shared_ratio
        unique_bytes = attn_bytes - shared_bytes

        gpu_flops = 0.0
        gpu_bytes = 0.0
        pim_flops = 0.0
        pim_bytes = 0.0
        pcie_bytes = 0.0
        staged_bytes = 0.0
        spill_bytes = 0.0
        t_attn_gpu = 0.0
        t_attn_pim = 0.0
        t_moe_gpu = 0.0
        t_moe_pim = 0.0
        generate_time = 0.0
        moe_split_fraction = 1.0 if policy == "baseline_gpu" else 0.0
        moe_split_index = 0
        moe_activated_experts = 0
        moe_fanin_threshold = 0
        route_gpu_shared_attn = 0.0
        route_pim_shared_attn = 0.0
        route_gpu_moe = 0.0
        route_pim_moe = 0.0
        alpha_shared = 0.0
        t_gpu_attn_shared_stage = 0.0
        t_pim_attn_shared_stage = 0.0
        attn_residual_gap = 0.0
        unique_rebalance_triggered = False
        gpu_unique_paths = 0
        gpu_unique_path_ids = []
        t_gpu_attn_final = 0.0
        t_pim_attn_final = 0.0
        tech1_total_makespan_est = 0.0

        if policy in ["baseline_gpu", "baseline_gpu_sw"]:
            gpu_flops = attn_flops + moe_flops
            gpu_bytes = attn_bytes
            merge_bytes = 0.0
            control_bytes = 0.0
            t_attn_gpu = max(attn_flops / self.hw.cp_gpu, attn_bytes / self.hw.bw_gpu)
            t_moe_gpu, split_info = self._moe_all_gpu_time(step)
            if policy == "baseline_gpu_sw":
                attn_factor, moe_factor = self._gpu_sw_efficiency(step, action, shared_len, max(step.get("ctx", {}).get("ctx_len_mean", 0), 1))
                t_attn_gpu *= attn_factor
                t_moe_gpu *= moe_factor
            route_gpu_shared_attn = self._gpu_attention_time(shared_flops, shared_bytes)
            route_gpu_moe = t_moe_gpu
            moe_split_fraction = split_info["split_fraction"]
            moe_split_index = split_info["split_index"]
            moe_activated_experts = split_info["activated_experts"]
            moe_fanin_threshold = split_info["fanin_threshold"]
            generate_time = t_attn_gpu + t_moe_gpu
        elif policy in ["baseline_static", "ours"]:
            gpu_flops = moe_flops
            pim_flops = attn_flops
            pim_bytes = attn_bytes
            base_exchange = wk["n_token_ops"] * self.model["hidden_size"] * 2 * 2
            if policy == "baseline_static":
                # Static mapping has heavier control/metadata synchronization.
                pcie_bytes += base_exchange * 1.25
            else:
                pcie_bytes += base_exchange * 0.7
            route_pim_shared_attn = self._estimate_pim_attention_time(
                step=step,
                action=action,
                candidate=max(step.get("candidate_seqs", 0), 1),
                ctx_len=max(step.get("ctx", {}).get("ctx_len_mean", 0), 1),
                shared_len=shared_len,
                unique_len=0,
                alpha=0.0,
                shared_ratio=shared_ratio,
                pim_attn_bytes=shared_bytes,
                pim_attn_flops=shared_flops,
            )

            if policy == "ours" and (tech1 or self.orches):
                alpha, gamma, detail = self._search_alpha_gamma(
                    step=step,
                    action=action,
                    candidate=max(step.get("candidate_seqs", 0), 1),
                    ctx_len=max(step.get("ctx", {}).get("ctx_len_mean", 0), 1),
                    shared_len=shared_len,
                    unique_len=unique_len,
                    shared_ratio=shared_ratio,
                    shared_flops=shared_flops,
                    shared_bytes=shared_bytes,
                    unique_flops=unique_flops,
                    unique_bytes=unique_bytes,
                    moe_flops=moe_flops,
                    merge_bytes=merge_bytes,
                    alpha_prev=alpha_prev,
                    gamma_prev=gamma_prev,
                )
                gpu_flops = detail["gpu_flops"]
                gpu_bytes = detail["gpu_bytes"]
                pim_flops = detail["pim_flops"]
                pim_bytes = detail["pim_bytes"]
                spill_bytes = detail["spill_bytes"]
                t_attn_gpu = detail["t_attn_gpu"]
                t_attn_pim = detail["t_attn_pim"]
                t_moe_gpu = detail["t_moe_gpu"]
                t_moe_pim = detail["t_moe_pim"]
                generate_time = detail["t_generate"]
                moe_split_fraction = detail.get("moe_split_fraction", 0.0)
                moe_split_index = detail.get("moe_split_index", 0)
                moe_activated_experts = detail.get("moe_activated_experts", 0)
                moe_fanin_threshold = detail.get("moe_fanin_threshold", 0)
                route_gpu_shared_attn = detail.get("t_gpu_attn_shared_only", 0.0)
                route_pim_shared_attn = detail.get("t_pim_attn_shared_only", 0.0)
                route_gpu_moe = t_moe_gpu
                route_pim_moe = t_moe_pim
                alpha_shared = detail.get("alpha_shared", alpha)
                t_gpu_attn_shared_stage = detail.get("t_gpu_attn_shared_stage", t_attn_gpu)
                t_pim_attn_shared_stage = detail.get("t_pim_attn_shared_stage", t_attn_pim)
                attn_residual_gap = detail.get("attn_residual_gap", t_pim_attn_shared_stage - t_gpu_attn_shared_stage)
                unique_rebalance_triggered = detail.get("unique_rebalance_triggered", False)
                gpu_unique_paths = detail.get("gpu_unique_paths", 0)
                gpu_unique_path_ids = detail.get("gpu_unique_path_ids", [])
                t_gpu_attn_final = detail.get("t_gpu_attn_final", t_attn_gpu)
                t_pim_attn_final = detail.get("t_pim_attn_final", t_attn_pim)
                tech1_total_makespan_est = detail.get("tech1_total_makespan_est", generate_time)
                staged_bytes = abs(alpha - alpha_prev) * shared_bytes * 0.02
                pcie_bytes += staged_bytes
                pcie_bytes += spill_bytes
                pcie_bytes *= 0.80 if self.orches else 0.62
            else:
                t_attn_pim = self._estimate_pim_attention_time(
                    step=step,
                    action=action,
                    candidate=max(step.get("candidate_seqs", 0), 1),
                    ctx_len=max(step.get("ctx", {}).get("ctx_len_mean", 0), 1),
                    shared_len=shared_len,
                    unique_len=unique_len,
                    alpha=0.0,
                    shared_ratio=shared_ratio,
                    pim_attn_bytes=pim_bytes,
                    pim_attn_flops=pim_flops,
                )
                t_moe_gpu, split_info = self._moe_all_gpu_time(step)
                route_gpu_moe = t_moe_gpu
                route_pim_shared_attn = self._estimate_pim_attention_time(
                    step=step,
                    action=action,
                    candidate=max(step.get("candidate_seqs", 0), 1),
                    ctx_len=max(step.get("ctx", {}).get("ctx_len_mean", 0), 1),
                    shared_len=shared_len,
                    unique_len=0,
                    alpha=0.0,
                    shared_ratio=shared_ratio,
                    pim_attn_bytes=shared_bytes,
                    pim_attn_flops=shared_flops,
                )
                moe_split_fraction = split_info["split_fraction"]
                moe_split_index = split_info["split_index"]
                moe_activated_experts = split_info["activated_experts"]
                moe_fanin_threshold = split_info["fanin_threshold"]
                generate_time = max(t_attn_gpu, t_attn_pim) + max(t_moe_gpu, t_moe_pim)
        else:
            raise ValueError(f"Unsupported policy: {policy}")

        t_gpu = max(gpu_flops / self.hw.cp_gpu, gpu_bytes / self.hw.bw_gpu)
        pim_bw = self.hw.bw_pim_int * (1.08 if self.fast_pim_model else 1.0)
        t_pim = max(pim_flops / self.hw.cp_pim, pim_bytes / pim_bw)
        if not self.fast_pim_model and pim_bytes > 0 and self.pim_runtime is not None:
            t_pim_attn = self._estimate_pim_attention_runtime(
                step=step,
                action=action,
                candidate=max(step.get("candidate_seqs", 0), 1),
                ctx_len=max(step.get("ctx", {}).get("ctx_len_mean", 0), 1),
                shared_len=shared_len,
                unique_len=unique_len,
                alpha=alpha,
                shared_ratio=shared_ratio,
                pim_bytes=pim_bytes,
            )
            t_pim = t_pim_attn + t_moe_pim
            if not tech1 or policy != "ours":
                t_attn_pim = t_pim_attn
                t_moe_pim = 0.0
        if policy in ["baseline_gpu", "baseline_gpu_sw"]:
            t_gpu = t_attn_gpu + t_moe_gpu
            t_pim = 0.0
        elif policy in ["baseline_static", "ours"]:
            t_gpu = t_attn_gpu + t_moe_gpu
            t_pim = t_attn_pim + t_moe_pim
        merge_ms = self._to_ms(merge_bytes / self.hw.bw_link)
        comm_ms = self._to_ms((pcie_bytes + merge_bytes + control_bytes) / self.hw.bw_link)

        return {
            "action": action,
            "alpha": alpha,
            "gamma": gamma,
            "gpu_flops": gpu_flops,
            "pim_flops": pim_flops,
            "gpu_bytes": gpu_bytes,
            "pim_bytes": pim_bytes,
            "merge_bytes": merge_bytes,
            "pcie_bytes": pcie_bytes,
            "control_bytes": control_bytes,
            "staged_bytes": staged_bytes,
            "spill_bytes": spill_bytes,
            "gpu_compute_ms": self._to_ms(t_gpu),
            "pim_compute_ms": self._to_ms(t_pim),
            "generate_ms": self._to_ms(generate_time if generate_time > 0 else max(t_gpu, t_pim)),
            "merge_ms": merge_ms,
            "pcie_comm_ms": comm_ms,
            "moe_split_fraction": moe_split_fraction,
            "moe_split_index": moe_split_index,
            "moe_activated_experts": moe_activated_experts,
            "moe_fanin_threshold": moe_fanin_threshold,
            "alpha_shared": alpha_shared,
            "t_gpu_attn_shared_stage_ms": self._to_ms(t_gpu_attn_shared_stage),
            "t_pim_attn_shared_stage_ms": self._to_ms(t_pim_attn_shared_stage),
            "attn_residual_gap_ms": self._to_ms(attn_residual_gap),
            "unique_rebalance_triggered": unique_rebalance_triggered,
            "gpu_unique_paths": gpu_unique_paths,
            "gpu_unique_path_ids": gpu_unique_path_ids,
            "t_gpu_attn_final_ms": self._to_ms(t_gpu_attn_final),
            "t_pim_attn_final_ms": self._to_ms(t_pim_attn_final),
            "tech1_total_makespan_est_ms": self._to_ms(tech1_total_makespan_est),
            "route_gpu_shared_attn_ms": self._to_ms(route_gpu_shared_attn),
            "route_pim_shared_attn_ms": self._to_ms(route_pim_shared_attn),
            "route_gpu_moe_ms": self._to_ms(route_gpu_moe),
            "route_pim_moe_ms": self._to_ms(route_pim_moe),
        }

    def _estimate_pim_attention_runtime(self,
                                        step: dict,
                                        action: str,
                                        candidate: int,
                                        ctx_len: int,
                                        shared_len: int,
                                        unique_len: int,
                                        alpha: float,
                                        shared_ratio: float,
                                        pim_bytes: float) -> float:
        token_len = max(step.get("tokens_generated_per_seq", 0), 1)
        kv_heads = self.model["num_key_value_heads"]
        head_dim = self.model["head_dim"]
        layers = self.model["num_hidden_layers"]
        hidden = self.model["hidden_size"]

        base_kv_bytes_per_token = 2 * kv_heads * head_dim * 2 * layers
        if action in ["multitok", "branch"]:
            shared_scale = max(0.0, min(shared_ratio * (1.0 - alpha), 1.0))
            unique_scale = max(0.0, min(1.0 - shared_ratio, 1.0))
            total_seconds = 0.0
            if unique_scale > 0:
                tail_desc = PIMAttentionDescriptor(
                    op_type="tail_attn",
                    num_paths=candidate,
                    seq_len=max(unique_len, 1),
                    ctx_len=max(ctx_len, 1),
                    token_len=token_len,
                    kv_heads=kv_heads,
                    head_dim=head_dim,
                    num_layers=layers,
                    hidden_size=hidden,
                )
                total_seconds += self.pim_runtime.run_attention(
                    tail_desc, scale=unique_scale)["latency_ns"] * 1e-9
            if shared_scale > 0:
                shared_desc = PIMAttentionDescriptor(
                    op_type="shared_attn",
                    num_paths=candidate,
                    seq_len=max(shared_len, 1),
                    ctx_len=max(ctx_len, 1),
                    token_len=token_len,
                    kv_heads=kv_heads,
                    head_dim=head_dim,
                    num_layers=layers,
                    hidden_size=hidden,
                )
                total_seconds += self.pim_runtime.run_attention(
                    shared_desc, scale=shared_scale)["latency_ns"] * 1e-9
            return total_seconds

        # Decode/singletok fall back to a decode-style attention replay.
        full_ctx_desc = PIMAttentionDescriptor(
            op_type="decode_attn",
            num_paths=candidate,
            seq_len=max(ctx_len, 1),
            ctx_len=max(ctx_len, 1),
            token_len=token_len,
            kv_heads=kv_heads,
            head_dim=head_dim,
            num_layers=layers,
            hidden_size=hidden,
        )
        desc_bytes = candidate * token_len * ctx_len * base_kv_bytes_per_token
        scale = max(pim_bytes / max(desc_bytes, 1.0), 1e-6)
        return self.pim_runtime.run_attention(full_ctx_desc, scale=scale)["latency_ns"] * 1e-9
