"""Controller mapper computation cost, separate from installation/migration.

This is the paper's operation-count estimate at 200 MHz, not a Python host
timing or an RTL measurement.  The attention search reserves five candidate
slots per invocation; the MoE sort/prefix scan scales with activated experts.
"""

from __future__ import annotations

import math


MAPPER_CLOCK_MHZ = 200
# Five candidate slots plus this fixed setup retain the archived figure's
# 60-cycle attention-search budget.  The old analysis path used six slots and
# a 24-cycle setup, contradicting the paper's five-candidate method.
ATTENTION_SETUP_CYCLES = 30
ATTENTION_CYCLES_PER_ALPHA = 6
DEFAULT_ALPHA_CANDIDATES = 5


def mapping_computation_charge(activated_experts: int,
                               alpha_candidates: int = DEFAULT_ALPHA_CANDIDATES,
                               *, invoked: bool = True) -> dict:
    """Return computation-only charge; no install, staging, DMA or KV movement."""
    if activated_experts < 0 or alpha_candidates < 1:
        raise ValueError("Invalid mapper problem size")
    if not invoked:
        attn_cycles = moe_cycles = 0
    else:
        attn_cycles = ATTENTION_SETUP_CYCLES + (
            ATTENTION_CYCLES_PER_ALPHA * alpha_candidates)
        moe_cycles = 2 * activated_experts + 4 * math.ceil(
            math.log2(activated_experts + 1))
    total_cycles = attn_cycles + moe_cycles
    return {
        "mapper_invoked": bool(invoked),
        "activated_experts": activated_experts if invoked else 0,
        "alpha_candidate_slots": alpha_candidates if invoked else 0,
        "attention_cycles": attn_cycles,
        "moe_cycles": moe_cycles,
        "mapping_cycles": total_cycles,
        "attention_us": attn_cycles / MAPPER_CLOCK_MHZ,
        "moe_us": moe_cycles / MAPPER_CLOCK_MHZ,
        "mapping_us": total_cycles / MAPPER_CLOCK_MHZ,
        "attention_ms": attn_cycles / (MAPPER_CLOCK_MHZ * 1000.0),
        "moe_ms": moe_cycles / (MAPPER_CLOCK_MHZ * 1000.0),
        "mapping_ms": total_cycles / (MAPPER_CLOCK_MHZ * 1000.0),
    }
