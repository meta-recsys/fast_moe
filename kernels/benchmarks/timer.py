# pyre-strict
import abc
import ctypes
import math

import time
from dataclasses import dataclass
from typing import Callable, cast, List, Optional

import pandas as pd
import torch
from scipy import stats


# Loop function result type
class TimerScale:
    SEC: float = 1.0
    MILLI: float = 1e-03
    MICRO: float = 1e-06
    NANO: float = 1e-09


# Timer result with 1 microsecond maximum granularity
@dataclass(frozen=True)
class TimerResult:
    sample_size: int  # number of recorded samples
    mean_sec: float  # mean result value
    corrected_mean_sec: float  # mean result value with outliers filtered out
    median_sec: float  # median result value aka p50
    stddev_sec: float  # standard deviation
    stderr_sec: float  # standard error
    relerr_percent: float  # relative error (standard deviation over mean, in %)
    p75_sec: float  # 75th percentile
    p90_sec: float  # 90th percentile
    p95_sec: float  # 95th percentile


@dataclass(frozen=True)
class TimerParams:
    warmup_iters: int = 10
    flush_gpu_cache_size_mb: int = 40
    lower_percentile_threshold: float = 0.2
    upper_percentile_threshold: float = 0.8

    def __post_init__(self) -> None:
        assert (
            self.warmup_iters >= 0
            and self.flush_gpu_cache_size_mb >= 0
            and self.lower_percentile_threshold >= 0.0
            and self.lower_percentile_threshold <= 1.0
            and self.upper_percentile_threshold >= 0.0
            and self.upper_percentile_threshold <= 1.0
            and self.lower_percentile_threshold < self.upper_percentile_threshold
        )


@dataclass(frozen=True)
class LegacyTimerParams(TimerParams):
    skip_first: int = 10
    initial_est_iters: int = 100
    confidence_interval: float = 0.95
    margin_error: float = 0.05

    def __post_init__(self) -> None:
        super().__post_init__()
        assert (
            self.skip_first >= 0
            and self.initial_est_iters > 0
            and self.confidence_interval > 0.0
            and self.confidence_interval < 1.0
            and self.margin_error > 0.0
        )


@dataclass(frozen=True)
class StableTimerParams(TimerParams):
    total_time_sec: float = 120.0
    epochs: int = 500

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.total_time_sec > 0.0 and self.epochs > 0


# Approximates the sample size necessary to achieve
# conf_int confidence and marg_err margin of error
# https://www.itl.nist.gov/div898/handbook/prc/section2/prc222.htm
def approx_sample_size(
    conf_int: float, marg_err: float, mean: float, std_dev: float
) -> int:
    q_val = 1.0 - (1.0 - conf_int) / 2
    z_val = stats.norm.ppf(q=q_val)
    n_approx = int(((z_val * std_dev) / (marg_err * mean)) ** 2) + 1
    if n_approx == 1:
        return n_approx
    t_val = stats.t.ppf(q=q_val, df=n_approx - 1)
    n_approx = int(((t_val * std_dev) / (marg_err * mean)) ** 2) + 1
    return n_approx


# Timer class
class Timer(abc.ABC):
    def __init__(
        self, overhead_sec: Optional[float] = None, params: Optional[TimerParams] = None
    ) -> None:
        self._params: TimerParams = self.cast_timer_params(params)
        if overhead_sec is not None:
            self._overhead_sec: float = overhead_sec
        else:
            # Computes the overhead by timing lambda: None with zero overhead
            self._overhead_sec: float = 0.0
            self._overhead_sec: float = self.time(func=lambda: None).median_sec

    # Downcasts TimerParams to LegacyTimerParams or StableTimerParams, if necessary
    # or creates them, if necessary
    @abc.abstractmethod
    def cast_timer_params(self, params: Optional[TimerParams]) -> TimerParams: ...

    @abc.abstractmethod
    def initialize_timing(self) -> None:
        pass

    # Run func in loop for cnt iters and provide approximate time per iteration in seconds
    @abc.abstractmethod
    def run_iters(self, func: Callable[[], None], cnt: int) -> float: ...

    # Compute TimerResult from measurements in seconds
    def compute_results(self, measurements_sec: List[float]) -> TimerResult:
        n = len(measurements_sec)
        d = pd.DataFrame({"t": measurements_sec})
        d.sort_values("t", inplace=True)

        return TimerResult(
            sample_size=n,
            # pyre-ignore[16]
            mean_sec=d.mean().t,
            corrected_mean_sec=d[
                int(n * self._params.lower_percentile_threshold) : int(
                    n * self._params.upper_percentile_threshold
                )
            ]
            .mean()
            .t,
            median_sec=float(d["t"].quantile(0.50)),
            # pyre-ignore[16]
            stddev_sec=d.std().t,
            # pyre-ignore[16]
            stderr_sec=d.std().t / math.sqrt(len(d)),
            # pyre-ignore[16]
            relerr_percent=d.std().t / d.mean().t * 100,
            p75_sec=float(d["t"].quantile(0.75)),
            p90_sec=float(d["t"].quantile(0.9)),
            p95_sec=float(d["t"].quantile(0.95)),
        )

    # Time func
    @abc.abstractmethod
    def time(self, func: Callable[[], None]) -> TimerResult: ...


class CPUTimer(Timer):
    def run_iters(self, func: Callable[[], None], cnt: int) -> float:
        start_time_ns = time.time_ns()
        for _ in range(cnt):
            func()
        elapsed_time_ns = time.time_ns() - start_time_ns
        return (elapsed_time_ns * TimerScale.NANO - self._overhead_sec) / cnt


class GPUTimer(Timer):
    def initialize_timing(self) -> None:
        if self._params.flush_gpu_cache_size_mb > 0:
            torch.cuda.synchronize()
            _ = torch.rand(
                self._params.flush_gpu_cache_size_mb
                * 1024
                * 1024
                // ctypes.sizeof(ctypes.c_float),
                dtype=torch.float,
            )
            torch.cuda.synchronize()

    def run_iters(self, func: Callable[[], None], cnt: int) -> float:
        assert torch.cuda.is_available()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(cnt):
            func()
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        return (elapsed_time_ms * TimerScale.MILLI - self._overhead_sec) / cnt


class LegacyTimer(Timer):
    def cast_timer_params(self, params: Optional[TimerParams]) -> TimerParams:
        if params is None:
            return LegacyTimerParams()
        if isinstance(params, LegacyTimerParams):
            return params
        else:
            return LegacyTimerParams(
                warmup_iters=params.warmup_iters,
                flush_gpu_cache_size_mb=params.flush_gpu_cache_size_mb,
                lower_percentile_threshold=params.lower_percentile_threshold,
                upper_percentile_threshold=params.upper_percentile_threshold,
            )

    def _time_func(self, func: Callable[[], None], active_iters: int) -> TimerResult:
        params: LegacyTimerParams = cast(LegacyTimerParams, self._params)
        if params.warmup_iters > 0:
            self.run_iters(func, params.warmup_iters)
        self.initialize_timing()
        measurements_sec = []
        if params.skip_first > 0:
            self.run_iters(func, params.skip_first)
        for _ in range(active_iters):
            measurements_sec.append(self.run_iters(func, 1))
        return self.compute_results(measurements_sec)

    def time(self, func: Callable[[], None]) -> TimerResult:
        """
        Given function `run_iters` that runs benchmark requested number of times
        and returns a single iteration timing in seconds we do the following:
        - Run the function for initial_est_iters and time each iteration
        - Use the mean, standard deviation from the above data to approximate
          the number of iterations necessary to achieve confidence_interval and margin_error
        - Run the function for approx_iters and time each iteration and return the results
        """
        params: LegacyTimerParams = cast(LegacyTimerParams, self._params)
        initial_results = self._time_func(
            func=func, active_iters=params.initial_est_iters
        )

        return initial_results


class StableTimer(Timer):
    def cast_timer_params(self, params: Optional[TimerParams]) -> TimerParams:
        if params is None:
            return StableTimerParams()
        if isinstance(params, StableTimerParams):
            return params
        else:
            return StableTimerParams(
                warmup_iters=params.warmup_iters,
                flush_gpu_cache_size_mb=params.flush_gpu_cache_size_mb,
                lower_percentile_threshold=params.lower_percentile_threshold,
                upper_percentile_threshold=params.upper_percentile_threshold,
            )

    def time(self, func: Callable[[], None]) -> TimerResult:
        """Accurate time estimation function benchmarking primitive

        Given function `run_iters` that runs benchmark requested number of times
        and returns a single iteration timing in seconds we do the following:
        - Run once and estimate time
        - Run for 1sec to get better estimation
        - Calculate epoch time `total_time / epochs`
        - Run `epochs` tests with estimated number of iterations and return results
        """
        params: StableTimerParams = cast(StableTimerParams, self._params)

        if params.warmup_iters > 0:
            self.run_iters(func, params.warmup_iters)

        self.initialize_timing()

        # Estimate running time for a single iteration
        single_run_time_sec = self.run_iters(func, 1)
        # Do better estimate over 1sec run
        iters_per_sec = 1.0 / single_run_time_sec
        single_run_time_sec = self.run_iters(func, int(iters_per_sec))

        # Now we know quite good approx for a single iteration time
        # Split total time into epochs
        num_iters_per_epoch = int(
            (params.total_time_sec / float(params.epochs)) / single_run_time_sec
        )
        num_iters_per_epoch = max(1, num_iters_per_epoch)

        measurements_sec: List[float] = []
        for _ in range(params.epochs):
            measurements_sec.append(self.run_iters(func, num_iters_per_epoch))

        return self.compute_results(measurements_sec)


class LegacyCPUTimer(LegacyTimer, CPUTimer): ...


class LegacyGPUTimer(LegacyTimer, GPUTimer): ...


class StableCPUTimer(StableTimer, CPUTimer): ...


class StableGPUTimer(StableTimer, GPUTimer): ...
