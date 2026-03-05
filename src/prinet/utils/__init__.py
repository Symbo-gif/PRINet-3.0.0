"""PRINet Utilities and GPU Kernels.

Provides GPU-accelerated ODE solvers, sparse coupling matrix utilities,
gradient checkpointing for memory-efficient training, fused Triton
kernels for Kuramoto oscillator dynamics, and CUDA C++ fused kernels.

Modules:
    cuda_kernels: BatchedRK45Solver, FixedStepRK4Solver, sparse utilities.
    triton_kernels: Fused Triton kernels for mean-field RK4 and sparse k-NN.
    fused_kernels: CUDA C++ fused discrete step, mixed-precision, CSR sparse,
        large-scale oscillator systems, async pipeline, and pruning.
"""

from prinet.utils.benchmark_reporting import (
    generate_benchmark_report,
    generate_leaderboard,
    generate_scalr_metrics_report,
)
from prinet.utils.cuda_kernels import (
    BatchedRK45Solver,
    FixedStepRK4Solver,
    SolverResult,
    gradient_checkpoint_integration,
    sparse_coupling_matrix,
)
from prinet.utils.fused_kernels import (
    AsyncCPUGPUPipeline,
    LargeScaleOscillatorSystem,
    MixedPrecisionTrainer,
    OscillatorPruner,
    build_knn_neighbors,
    csr_coupling_step,
    cuda_fused_kernel_available,
    fused_discrete_step_cuda,
    pytorch_fused_discrete_step_full,
    sparse_coupling_matrix_csr,
    sparse_knn_coupling_step,
)
from prinet.utils.npu_backend import (
    BackendType,
    backend_info,
    create_session,
    detect_best_backend,
    directml_available,
    npu_available,
)
from prinet.utils.oscillosim import (
    OscilloSim,
    SimulationResult,
    bimodality_index,
    local_order_parameter,
    quick_simulate,
    ring_topology,
    small_world_topology,
)
from prinet.utils.temporal_metrics import (
    TemporalMetrics,
    binding_robustness_score,
    compute_full_temporal_metrics,
    identity_overcount,
    identity_switches,
    mostly_tracked_lost,
    recovery_speed,
    temporal_smoothness,
    track_duration_stats,
    track_fragmentation_rate,
)
from prinet.utils.temporal_training import (
    MultiSeedResult,
    SequenceData,
    TemporalTrainer,
    TrainingResult,
    TrainingSnapshot,
    count_parameters,
    generate_dataset,
    generate_temporal_clevr_n,
    hungarian_similarity_loss,
    temporal_smoothness_loss,
    train_multi_seed,
)
from prinet.utils.triton_kernels import (
    pytorch_cross_band_coupling,
    pytorch_fused_discrete_step,
    pytorch_fused_sub_step_rk4,
    pytorch_hierarchical_order_param,
    pytorch_mean_field_rk4_step,
    pytorch_multi_rate_derivatives,
    pytorch_multi_rate_rk4_step,
    pytorch_pac_modulation,
    pytorch_sparse_knn_coupling,
    triton_available,
    triton_fused_discrete_step,
    triton_fused_mean_field_rk4_step,
    triton_hierarchical_order_param,
    triton_pac_modulation,
    triton_sparse_knn_coupling,
)
from prinet.utils.y4q1_tools import (
    AblationConfig,
    AblationHybridPRINetV2,
    ExtendedTrainingResult,
    binding_persistence,
    coherence_decay_rate,
    compute_p_value,
    count_flops,
    create_ablation_model,
    cross_frequency_coupling,
    cumulative_phase_slip_curve,
    instantaneous_frequency_spread,
    measure_wall_time,
    memory_growth_profile,
    order_parameter_series,
    phase_locking_value,
    phase_slip_rate,
    rebinding_speed,
    session_length_statistical_comparison,
    temporal_advantage_report,
    throughput_series,
    train_clevr_n_extended,
    train_clevr_n_single_seed,
    windowed_order_parameter_variance,
)

__all__ = [
    "BatchedRK45Solver",
    "FixedStepRK4Solver",
    "SolverResult",
    "sparse_coupling_matrix",
    "gradient_checkpoint_integration",
    # Triton kernels (Q2)
    "triton_available",
    "triton_fused_mean_field_rk4_step",
    "triton_sparse_knn_coupling",
    "pytorch_mean_field_rk4_step",
    "pytorch_sparse_knn_coupling",
    # Triton kernels (Q3)
    "triton_pac_modulation",
    "triton_hierarchical_order_param",
    "pytorch_pac_modulation",
    "pytorch_hierarchical_order_param",
    "pytorch_multi_rate_rk4_step",
    # Triton kernels (Q3 late — multi-rate)
    "pytorch_multi_rate_derivatives",
    "pytorch_fused_sub_step_rk4",
    "pytorch_cross_band_coupling",
    # Y2 Q3: Fused discrete recurrence
    "triton_fused_discrete_step",
    "pytorch_fused_discrete_step",
    # Y3 Q3: CUDA C++ fused kernels
    "cuda_fused_kernel_available",
    "fused_discrete_step_cuda",
    "pytorch_fused_discrete_step_full",
    # Y3 Q3: Mixed-precision training (O.3)
    "MixedPrecisionTrainer",
    # Y3 Q3: CSR sparse coupling (R.4)
    "sparse_coupling_matrix_csr",
    "csr_coupling_step",
    # Y3 Q3: Large-scale oscillators (O.4)
    "LargeScaleOscillatorSystem",
    "build_knn_neighbors",
    "sparse_knn_coupling_step",
    # Y3 Q3: Async CPU+GPU pipeline (O.5)
    "AsyncCPUGPUPipeline",
    # Y3 Q3: Model pruning (O.6)
    "OscillatorPruner",
    # Benchmark reporting (Q3)
    "generate_benchmark_report",
    "generate_leaderboard",
    "generate_scalr_metrics_report",
    # NPU backend (Q4)
    "BackendType",
    "detect_best_backend",
    "npu_available",
    "directml_available",
    "create_session",
    "backend_info",
    # Y3 Q4: OscilloSim v2.0 (P.4)
    "OscilloSim",
    "SimulationResult",
    "quick_simulate",
    # Y4 Q1: Ring/small-world topologies & chimera detection (T.1, T.2)
    "ring_topology",
    "small_world_topology",
    "local_order_parameter",
    "bimodality_index",
    # Y4 Q1: Ablation, extended training, FLOPs (T.4, T.5, T.6)
    "AblationConfig",
    "AblationHybridPRINetV2",
    "create_ablation_model",
    "ExtendedTrainingResult",
    "train_clevr_n_single_seed",
    "train_clevr_n_extended",
    "compute_p_value",
    "count_flops",
    "measure_wall_time",
    # Y4 Q1.4: Temporal advantage metrics
    "phase_slip_rate",
    "binding_persistence",
    "coherence_decay_rate",
    "rebinding_speed",
    "cross_frequency_coupling",
    "temporal_advantage_report",
    # Y4 Q1.5: Session-length metrics
    "order_parameter_series",
    "windowed_order_parameter_variance",
    "phase_locking_value",
    "instantaneous_frequency_spread",
    "cumulative_phase_slip_curve",
    "throughput_series",
    "memory_growth_profile",
    "session_length_statistical_comparison",
    # Y4 Q1.7: Temporal metrics
    "TemporalMetrics",
    "temporal_smoothness",
    "identity_switches",
    "track_fragmentation_rate",
    "identity_overcount",
    "mostly_tracked_lost",
    "track_duration_stats",
    "recovery_speed",
    "binding_robustness_score",
    "compute_full_temporal_metrics",
    # Y4 Q1.7: Temporal training
    "SequenceData",
    "generate_temporal_clevr_n",
    "generate_dataset",
    "hungarian_similarity_loss",
    "temporal_smoothness_loss",
    "count_parameters",
    "TrainingSnapshot",
    "TrainingResult",
    "TemporalTrainer",
    "MultiSeedResult",
    "train_multi_seed",
]
