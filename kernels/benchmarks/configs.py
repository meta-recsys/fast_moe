# pyre-strict
import dataclasses


@dataclasses.dataclass
class PT2Config:
    backend: str = "inductor"
    cudagraphs: bool = False
    dynamic_shapes: bool = False


@dataclasses.dataclass
class ProfilerParams:
    wait_cycles: int = 20
    warmup_cycles: int = 10
    active_cycles: int = 10
    memory_snapshot_cycles: int = 3
    gpu_only: bool = True
