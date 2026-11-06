class KVLifecycleModel:

    def __init__(self,
                 tech2: bool,
                 orches: bool = False,
                 orches_ab: bool = False,
                 bw_link: float = 64e9,
                 bw_pim_int: float = 242e12,
                 beta_threshold: float = 0.2,
                 bytes_per_block: int = 16 * 1024,
                 controller_cost_scale: float = 1.0):
        self.tech2 = tech2
        self.orches = orches
        self.orches_ab = orches_ab
        self.bw_link = bw_link
        self.bw_pim_int = bw_pim_int
        self.beta_threshold = beta_threshold
        self.bytes_per_block = bytes_per_block
        self.controller_cost_scale = controller_cost_scale
        self.live_blocks = None
        self.holes_blocks = None

    def _to_ms(self, seconds: float) -> float:
        return seconds * 1000.0

    def _init_state(self, step: dict):
        kv = step.get("kv", {})
        self.live_blocks = max(kv.get("kv_live_blocks", 0), 0)
        self.holes_blocks = max(kv.get("kv_holes_blocks", 0), 0)

    def _estimate_deltas(self, step: dict) -> tuple:
        kv = step.get("kv", {})
        action = step.get("action", "none")
        cand = max(step.get("candidate_seqs", 0), 0)
        alive = max(step.get("alive_paths", 0), 0)
        tok = max(step.get("tokens_generated_per_seq", 0), 0)

        alloc = max(kv.get("kv_alloc_blocks_delta", 0), 0)
        freed = max(kv.get("kv_free_blocks_delta", 0), 0)

        if action == "multitok":
            burst_alloc = max(cand * max(tok, 1) // 16, alloc)
            bulk_free = max((cand - alive) * max(tok, 1) // 16, freed)
            return burst_alloc, max(bulk_free, 0)
        if action == "branch":
            tail_growth = max(cand * max(tok, 1) // 24, alloc)
            gradual_prune = max((cand - alive) * max(tok, 1) // 48, freed)
            return tail_growth, max(gradual_prune, 0)
        return alloc, freed

    def handle_epoch(self, step: dict, policy: str = "baseline_static") -> dict:
        if self.live_blocks is None or self.holes_blocks is None:
            self._init_state(step)

        alloc_blocks, free_blocks = self._estimate_deltas(step)
        self.live_blocks = max(self.live_blocks + alloc_blocks - free_blocks, 0)
        self.holes_blocks = max(self.holes_blocks + free_blocks, 0)
        denom = max(self.live_blocks + self.holes_blocks, 1)
        beta = self.holes_blocks / denom

        alloc_events = 1 + (1 if step.get("action") in ["multitok", "branch"] else 0)
        moved_blocks = 0
        compacted_blocks = 0
        metadata_overhead_ms = 0.0
        pcie_bytes = 0.0

        total_blocks = max(self.live_blocks + self.holes_blocks, 1)
        if self.tech2:
            # HATT metadata sharing + PIM-side DMA movement.
            moved_blocks += int(free_blocks * 0.35)
            if beta > self.beta_threshold:
                compacted_blocks = int(self.holes_blocks * 0.5)
                moved_blocks += compacted_blocks
                self.holes_blocks = max(self.holes_blocks - int(compacted_blocks * 0.75), 0)
            moved_bytes = moved_blocks * self.bytes_per_block
            metadata_scan_bytes = total_blocks * self.bytes_per_block * 0.03
            kv_mgmt_ms = self._to_ms((moved_bytes + metadata_scan_bytes) / self.bw_pim_int)
            metadata_overhead_ms = self.controller_cost_scale * (0.0003 * alloc_events)
            kv_mgmt_ms += metadata_overhead_ms
        elif self.orches:
            # ORCHES only compacts holes into contiguous runs through software-managed movement.
            if beta > self.beta_threshold:
                compacted_blocks = int(self.holes_blocks * 0.35)
                moved_blocks += compacted_blocks
                self.holes_blocks = max(self.holes_blocks - int(compacted_blocks * 0.6), 0)
            moved_bytes = moved_blocks * self.bytes_per_block
            if self.orches_ab:
                # Stronger software-only action-aware baseline: reuse the
                # existing GPU-SW prefix aliasing and batched cleanup costs.
                # Compaction trigger and movement stay ORCHES-style, and no
                # HATT/controller-local metadata path is enabled.
                metadata_scan_bytes = total_blocks * self.bytes_per_block * 0.015
                gpu_alloc_overhead_ms = 0.65 * (
                    self.controller_cost_scale * (0.00045 * alloc_events)
                    + 0.0000015 * moved_blocks
                )
            else:
                metadata_scan_bytes = total_blocks * self.bytes_per_block * 0.16
                gpu_alloc_overhead_ms = self.controller_cost_scale * (0.0014 * alloc_events) + 0.000006 * moved_blocks
            moved_total = moved_bytes + metadata_scan_bytes
            kv_mgmt_ms = self._to_ms(moved_total / self.bw_link) + gpu_alloc_overhead_ms
            pcie_bytes = moved_total
        else:
            # Software-managed pool + link traffic + GPU allocator overhead.
            moved_blocks += free_blocks + int(alloc_blocks * 0.2)
            if beta > self.beta_threshold:
                compacted_blocks = int(self.holes_blocks * 0.5)
                moved_blocks += compacted_blocks
                self.holes_blocks = max(self.holes_blocks - int(compacted_blocks * 0.55), 0)
            moved_bytes = moved_blocks * self.bytes_per_block
            if policy == "baseline_gpu":
                # No GPU-PIM link in pure GPU baseline.
                gpu_alloc_overhead_ms = self.controller_cost_scale * (0.0009 * alloc_events) + 0.000004 * moved_blocks
                kv_mgmt_ms = gpu_alloc_overhead_ms
                pcie_bytes = 0.0
            elif policy == "baseline_gpu_sw":
                # Stronger software-only GPU baseline:
                # - paged KV allocation
                # - shared-prefix aliasing instead of deep prefix copy
                # - software-side reclaim and cleanup batching
                # - software-side compaction over SoC/GPU memory path
                action = step.get("action", "none")
                candidate = max(step.get("candidate_seqs", 0), 0)
                alive = max(step.get("alive_paths", 0), 0)
                shared_len = 0
                if action in ["multitok", "branch"]:
                    shared_len = max(step.get("ctx", {}).get("base_prompt_len", 0), 0)
                alias_factor = 0.30 if shared_len > 0 else 0.55
                batched_cleanup = 0.65
                reclaim_factor = 0.18

                moved_blocks = int((free_blocks * reclaim_factor + alloc_blocks * 0.03) * alias_factor)
                if beta > self.beta_threshold:
                    compacted_blocks = int(self.holes_blocks * 0.10)
                    moved_blocks += compacted_blocks
                    self.holes_blocks = max(self.holes_blocks - int(compacted_blocks * 0.85), 0)

                moved_bytes = moved_blocks * self.bytes_per_block
                metadata_scan_bytes = total_blocks * self.bytes_per_block * 0.015
                moved_total = moved_bytes + metadata_scan_bytes
                # Conservative software path: movement still consumes SoC/GPU memory bandwidth
                # and incurs a non-trivial metadata/allocator overhead, but it is materially
                # cheaper than the naive GPU-only baseline because prefix aliasing avoids
                # large copy amplification and cleanup is batched in software.
                gpu_alloc_overhead_ms = batched_cleanup * (self.controller_cost_scale * (0.00045 * alloc_events) + 0.0000015 * moved_blocks)
                kv_mgmt_ms = self._to_ms(moved_total / self.bw_link) + gpu_alloc_overhead_ms
                pcie_bytes = 0.0
            elif policy == "baseline_static":
                # Static mapping: software ownership and frequent synchronization.
                gpu_alloc_overhead_ms = self.controller_cost_scale * (0.0018 * alloc_events) + 0.000008 * moved_blocks
                metadata_scan_bytes = total_blocks * self.bytes_per_block * 0.35
                moved_total = moved_bytes * 3.0 + metadata_scan_bytes
                kv_mgmt_ms = self._to_ms(moved_total / self.bw_link) + gpu_alloc_overhead_ms
                pcie_bytes = moved_total
            else:
                gpu_alloc_overhead_ms = self.controller_cost_scale * (0.0012 * alloc_events) + 0.000004 * moved_blocks
                metadata_scan_bytes = total_blocks * self.bytes_per_block * 0.12
                moved_total = moved_bytes * 1.8 + metadata_scan_bytes
                kv_mgmt_ms = self._to_ms(moved_total / self.bw_link) + gpu_alloc_overhead_ms
                pcie_bytes = moved_total

        return {
            "kv_mgmt_ms": kv_mgmt_ms,
            "pcie_bytes": pcie_bytes,
            "moved_blocks": moved_blocks,
            "compacted_blocks": compacted_blocks,
            "kv_live_blocks": self.live_blocks,
            "kv_holes_blocks": self.holes_blocks,
            "beta": beta,
            "metadata_overhead_ms": metadata_overhead_ms,
        }
