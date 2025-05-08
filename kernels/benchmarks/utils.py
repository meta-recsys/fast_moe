# pyre-strict

import abc
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from fast_moe.kernels.benchmarks.configs import ProfilerParams, PT2Config
from fast_moe.kernels.benchmarks.timer import (
    LegacyGPUTimer,
    StableGPUTimer,
    TimerResult,
)
from fast_moe.kernels.utils import KernelType

from torch.cuda import nvtx
from torch.profiler import profile
from torch.utils.flop_counter import FlopCounterMode

str2dtype: Dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s:%(lineno)s] %(message)s")
# pyre-ignore
handler: logging.StreamHandler = logging.StreamHandler()
handler.setLevel(level=logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


@dataclass
class BenchmarkResult:
    timer_result: TimerResult
    memory_result: Dict[str, Any]


class suppress_stdout_stderr(object):
    """
    Define a context manager for temporarily suppressing stdout and stderr
    since pvc2.hive_to_pandas will produce massive output in the terminal.
    This will not suppress raised exceptions.
    """

    def __init__(self) -> None:
        # Open a pair of null files
        self.null_fds: List[int] = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds: List[int] = [os.dup(1), os.dup(2)]

    def __enter__(self) -> None:
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_: Tuple[Any]) -> None:
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def get_kernel(provider: str) -> KernelType:
    if provider in ["triton", "triton_split_k", "triton_split_k_tma"]:
        return KernelType.TRITON
    elif provider == "pytorch":
        return KernelType.PYTORCH
    else:
        raise ValueError(f"Unknown provider {provider}")


def print_timer_result(res: TimerResult, tag: str = "", verbose: bool = False) -> None:
    logger.info(f"{'(' + tag + ') ' if tag else ''}Benchmark Results: ")
    logger.info(f"\tSample Size: {res.sample_size : >11}")
    logger.info(f"\tMean: {res.mean_sec * 1e3 : >18.5} msec")
    logger.info(f"\tStd Dev: {res.stddev_sec * 1e3 : >15.5} msec")
    logger.info(f"\tRel Error: {res.relerr_percent : >13.5} %")
    if verbose:
        logger.info(f"\tCorr Mean: {res.corrected_mean_sec * 1e3 : >13.5} msec")
        logger.info(f"\tStd Error: {res.stderr_sec * 1e3 : >13.5} msec")
        logger.info(f"\tMedian: {res.median_sec * 1e3 : >16.5} msec")
        logger.info(f"\tp75: {res.p75_sec * 1e3 : >19.5} msec")
        logger.info(f"\tp90: {res.p90_sec * 1e3 : >19.5} msec")
        logger.info(f"\tp95: {res.p95_sec * 1e3 : >19.5} msec")


class ModuleFactory(abc.ABC):
    # Note: When making a subclass of ModuleFactory, add an __init__() to
    # pass in all necessary module configuration variables. See
    # user_item_embedding_inner_product_bench.py for an example.
    def __init__(self, config: object = None) -> None:
        self.config = config

    # Returns the name of the module in snakecase (i.e. module_name)
    @abc.abstractmethod
    def module_name(self) -> str:
        pass

    # Constructor for the module to be benchmarked
    @abc.abstractmethod
    def create_module(
        self,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.nn.Module:
        pass

    # Constructor for the arguments to be passed into the module to be benchmarked
    @abc.abstractmethod
    def create_inputs(
        self,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        pass

    # pyre-ignore
    def run_module(
        self,
        module: torch.nn.Module,
        inputs: Dict[str, Any],
    ) -> Any:
        return module(**inputs)


class ModuleBench(abc.ABC):
    def __init__(
        self,
        module_factory: ModuleFactory,
        fwd_func_name: str = "forward",
        timer_mode: str = "legacy",
        profiler_params: Optional[ProfilerParams] = None,
        enable_tf32: bool = True,
        precision: Optional[str] = None,
    ) -> None:
        assert torch.cuda.is_available()

        self._module_factory: ModuleFactory = module_factory
        self._fwd_func_name: str = fwd_func_name
        self._timer = StableGPUTimer() if timer_mode == "stable" else LegacyGPUTimer()

        self._profiler_params: ProfilerParams = (
            ProfilerParams() if profiler_params is None else profiler_params
        )

        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
        torch.backends.cudnn.allow_tf32 = enable_tf32

        self._module: torch.nn.Module = (
            self._module_factory.create_module(
                device=torch.device("cuda"), dtype=str2dtype[precision]
            )
            if precision is not None
            else self._module_factory.create_module(device=torch.device("cuda"))
        )

        self._module_inputs: Dict[str, Any] = (
            self._module_factory.create_inputs(
                device=torch.device("cuda"), dtype=str2dtype[precision]
            )
            if precision is not None
            else self._module_factory.create_inputs(device=torch.device("cuda"))
        )

    # Returns the benchmark type (i.e. "inference", "train", etc)
    @abc.abstractmethod
    def benchmark_type(self) -> str:
        pass

    # Subroutine to be benchmarked (i.e. forward pass, forward-backward pass)
    @abc.abstractmethod
    def run_module(self) -> None:
        pass

    def benchmark_module(self) -> TimerResult:
        return self._timer.time(self.run_module)

    def count_flops(self) -> int:
        counter = FlopCounterMode(display=False)
        with counter:
            self.run_module()
        return sum([v for _, v in counter.flop_counts["Global"].items()])

    def torch_profile(self) -> None:
        on_trace_ready = torch.profiler.tensorboard_trace_handler(
            dir_name="/tmp",
            worker_name=f"{self._module_factory.module_name()}-{self.benchmark_type().lower()}",
        )
        activities = [
            # pyre-fixme[16]: Module `profiler` has no attribute `ProfilerActivity`.
            torch.profiler.ProfilerActivity.CUDA,
        ]
        if not self._profiler_params.gpu_only:
            activities.append(
                # pyre-fixme[16]: Module `profiler` has no attribute `ProfilerActivity`.
                torch.profiler.ProfilerActivity.CPU,
            )
        profiler = profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=self._profiler_params.wait_cycles,
                warmup=self._profiler_params.warmup_cycles,
                active=self._profiler_params.active_cycles,
            ),
            on_trace_ready=on_trace_ready,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        )

        with suppress_stdout_stderr():
            nvtx_range_id = nvtx.range_start("benchmark.profile")
            profiler.start()
            for _ in range(
                self._profiler_params.wait_cycles
                + self._profiler_params.warmup_cycles
                + self._profiler_params.active_cycles
            ):
                self.run_module()
                profiler.step()
            nvtx.range_end(nvtx_range_id)
            profiler.stop()

        logger.info(
            "\n"
            + profiler.key_averages(group_by_input_shape=True).table(
                sort_by="self_cuda_time_total", row_limit=10
            )
        )

        for _ in range(self._profiler_params.warmup_cycles):
            self.run_module()
        for _ in range(self._profiler_params.memory_snapshot_cycles):
            self.run_module()

    def print_title(self) -> None:
        logger.info(
            "="
            * (
                len(self._module_factory.module_name())
                + len(self.benchmark_type())
                + 18
            )
        )
        logger.info(
            f"{self._module_factory.module_name().upper()} Module {self.benchmark_type().title()} Benchmark"
        )
        logger.info(
            "="
            * (
                len(self._module_factory.module_name())
                + len(self.benchmark_type())
                + 18
            )
        )

    def run_benchmark(
        self,
        measure_compute: bool = False,
        results_verbose: bool = False,
        enable_profiler: bool = True,
        return_result: bool = False,
    ) -> Optional[BenchmarkResult]:
        self.print_title()
        torch.cuda.reset_peak_memory_stats()

        results = self.benchmark_module()

        print_timer_result(results, verbose=results_verbose)

        if measure_compute:
            logger.info(f"Total Compute: {self.count_flops() / 1e9} gflops")

        if enable_profiler:
            self.torch_profile()

        if return_result:
            return BenchmarkResult(
                timer_result=results,
                memory_result=torch.cuda.memory_stats(),
            )


class TrainModuleBench(ModuleBench):
    def __init__(
        self,
        module_factory: ModuleFactory,
        fwd_func_name: str = "forward",
        timer_mode: str = "legacy",
        profiler_params: Optional[ProfilerParams] = None,
        enable_tf32: bool = True,
        precision: Optional[str] = None,
        amp_dtype: Optional[str] = None,
        pt2_config: Optional[PT2Config] = None,
        run_backward: bool = True,
    ) -> None:
        torch.set_grad_enabled(True)
        super().__init__(
            module_factory,
            fwd_func_name,
            timer_mode,
            profiler_params,
            enable_tf32,
            precision,
        )
        self._module.train()

        self._amp_dtype: Optional[torch.dtype] = (
            str2dtype[amp_dtype] if amp_dtype is not None else None
        )

        if pt2_config is not None:
            # pyre-ignore
            self.run_module = torch.compile(
                self.run_module,
                dynamic=pt2_config.dynamic_shapes,
                backend=pt2_config.backend,
                fullgraph=False,
                options={"triton.cudagraphs": pt2_config.cudagraphs},
            )
        self._run_backward = run_backward

    def benchmark_type(self) -> str:
        return "train"

    def run_module(self) -> None:
        func = getattr(self._module, self._fwd_func_name)
        retain_graph = (
            self._fwd_func_name != "forward"
        )  # if it is not forward, need to retain graph
        if self._amp_dtype is None:
            output = self._module_factory.run_module(func, self._module_inputs)
            if self._run_backward:
                loss = torch.randn_like(output)
                output.backward(loss, retain_graph=retain_graph)
        else:
            with torch.cuda.amp.autocast(dtype=self._amp_dtype):
                output = func(**self._module_inputs)
                if self._run_backward:
                    loss = output.sum()
                    loss.backward(retain_graph=retain_graph)
