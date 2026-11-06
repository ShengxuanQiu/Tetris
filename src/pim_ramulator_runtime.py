import json
import math
import pathlib
import subprocess
import sys
import tempfile
from dataclasses import dataclass


@dataclass(frozen=True)
class PIMAttentionDescriptor:
    op_type: str
    num_paths: int
    seq_len: int
    ctx_len: int
    token_len: int
    kv_heads: int
    head_dim: int
    num_layers: int
    hidden_size: int
    burst_bytes: int = 256

    def cache_key(self) -> str:
        return "|".join([
            self.op_type,
            str(self.num_paths),
            str(self.seq_len),
            str(self.ctx_len),
            str(self.token_len),
            str(self.kv_heads),
            str(self.head_dim),
            str(self.num_layers),
            str(self.hidden_size),
            str(self.burst_bytes),
        ])


class PIMTraceTranslator:

    def __init__(self,
                 trace_gen_root: pathlib.Path,
                 tile_seq_len: int = 256,
                 tile_head_cap: int = 32):
        self.trace_gen_root = trace_gen_root
        self.tile_seq_len = tile_seq_len
        self.tile_head_cap = tile_head_cap

    def _script_for_op(self, op_type: str) -> str:
        if op_type == "shared_attn":
            return "gen_trace_attacc_bank.py"
        if op_type == "tail_attn":
            return "gen_trace_attacc_bg.py"
        return "gen_trace_attacc_buffer.py"

    def _effective_len(self, desc: PIMAttentionDescriptor) -> int:
        if desc.op_type == "tail_attn":
            return max(desc.ctx_len + desc.seq_len, 1)
        if desc.op_type == "shared_attn":
            return max(desc.seq_len, 1)
        return max(desc.ctx_len, 1)

    def build_trace(self, desc: PIMAttentionDescriptor, trace_path: pathlib.Path) -> float:
        effective_len = self._effective_len(desc)
        sim_len = min(effective_len, self.tile_seq_len)
        sim_heads = min(max(desc.num_paths, 1), self.tile_head_cap)
        script = self.trace_gen_root / self._script_for_op(desc.op_type)

        cmd = [
            sys.executable,
            str(script),
            "-dh",
            str(desc.head_dim),
            "-nh",
            str(sim_heads),
            "-l",
            str(sim_len),
            "-maxl",
            str(max(effective_len, 4096)),
            "-db",
            "2",
            "-o",
            str(trace_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        length_tiles = math.ceil(effective_len / sim_len)
        head_tiles = math.ceil(max(desc.num_paths, 1) / sim_heads)
        query_tile = 256 if desc.op_type == "decode_attn" else 64
        token_tiles = math.ceil(max(desc.token_len, 1) / query_tile)
        return float(length_tiles * head_tiles * token_tiles * max(desc.num_layers, 1))


class PIMRamulatorRuntime:

    def __init__(self,
                 cache_path: str | None = None,
                 ramulator_root: str = "ramulator2",
                 tck_ns: float = 0.769):
        self.repo_root = pathlib.Path(__file__).resolve().parents[1]
        self.ramulator_root = self.repo_root / ramulator_root
        self.cache_path = pathlib.Path(cache_path) if cache_path else self.repo_root / "pim_ramulator_cache.json"
        self.tck_ns = tck_ns
        self.translator = PIMTraceTranslator(self.ramulator_root / "trace_gen")
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2, sort_keys=True)

    def _find_executable(self) -> pathlib.Path | None:
        candidates = [
            self.ramulator_root / "ramulator2",
            self.ramulator_root / "ramulator2.exe",
            self.ramulator_root / "build-linux" / "ramulator2",
            self.ramulator_root / "build-linux" / "ramulator2.exe",
            self.ramulator_root / "build" / "ramulator2",
            self.ramulator_root / "build" / "ramulator2.exe",
            self.ramulator_root / "build" / "Release" / "ramulator2.exe",
            self.ramulator_root / "build" / "Debug" / "ramulator2.exe",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _make_yaml(self, trace_path: pathlib.Path, yaml_path: pathlib.Path):
        content = "\n".join([
            "Frontend:",
            "  impl: PIMLoadStoreTrace",
            f"  path: {json.dumps(trace_path.as_posix(), ensure_ascii=False)}",
            "  clock_ratio: 1",
            "",
            "  Translation:",
            "    impl: NoTranslation",
            "    max_addr: 2147483648",
            "",
            "MemorySystem:",
            "  impl: PIMDRAM",
            "  clock_ratio: 1",
            "  DRAM:",
            "    impl: HBM3-PIM",
            "    org:",
            "      preset: HBM3_8Gb_2R",
            "      channel: 16",
            "    timing:",
            "      preset: HBM3_5.2Gbps",
            "",
            "  Controller:",
            "    impl: HBM3-PIM",
            "    Scheduler:",
            "      impl: PIM",
            "    RefreshManager:",
            "      impl: AllBankHBM3",
            "    plugins:",
            "",
            "  AddrMapper:",
            "    impl: HBM3-PIM",
            "",
        ])
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(content)

    def _parse_cycles(self, stdout: str) -> int:
        cycles = None
        for line in stdout.splitlines():
            if "memory_system_cycles" in line:
                parts = line.strip().split()
                cycles = int(parts[-1])
        if cycles is None:
            raise RuntimeError("Ramulator output did not contain memory_system_cycles.")
        return cycles

    def run_attention(self,
                      desc: PIMAttentionDescriptor,
                      scale: float = 1.0) -> dict:
        cache_key = f"tracegen-v2|{desc.cache_key()}|scale={scale:.6f}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        exe = self._find_executable()
        if exe is None:
            raise RuntimeError(
                "Ramulator2 executable not found. Run bash scripts/build_ramulator.sh."
            )

        with tempfile.TemporaryDirectory(prefix="pim_trace_", dir=str(self.repo_root)) as temp_dir:
            temp_dir_path = pathlib.Path(temp_dir)
            trace_path = temp_dir_path / "attn.trace"
            yaml_path = temp_dir_path / "attn.yaml"
            replay_scale = self.translator.build_trace(desc, trace_path)
            self._make_yaml(trace_path, yaml_path)

            cmd = [str(exe), "-f", str(yaml_path)]
            result = subprocess.run(cmd,
                                    cwd=str(self.ramulator_root),
                                    capture_output=True,
                                    text=True,
                                    check=True)
            base_cycles = self._parse_cycles(result.stdout)

        scaled_cycles = max(int(math.ceil(base_cycles * replay_scale * scale)), 1)
        latency_ns = scaled_cycles * self.tck_ns
        record = {
            "cycles": scaled_cycles,
            "latency_ns": latency_ns,
            "base_cycles": base_cycles,
            "replay_scale": replay_scale,
            "scale": scale,
            "op_type": desc.op_type,
        }
        self.cache[cache_key] = record
        self._save_cache()
        return record
