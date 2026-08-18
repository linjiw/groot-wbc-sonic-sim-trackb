"""Opt-in runtime accounting callback for one frozen LACE RQ1 arm."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from gear_sonic.research.lace.fixed_distribution import (
    FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD,
    assert_fixed_distribution_integrity,
    fixed_distribution_draw_report,
    validate_fixed_distribution_command_modes,
)
from gear_sonic.research.lace.rq1_training import (
    PLAN_DIGEST_FIELD,
    RQ1RuntimeAccounting,
    RQ1TrainingError,
    audit_saved_rq1_training_config,
    file_sha256,
    load_json_object,
    validate_attempt_claim,
    validate_initialization_report,
    validate_rq1_training_plan,
    validate_runtime_metrics,
    write_new_json,
)


class RQ1TrainingCallback(TrainerCallback):
    """Install exact accounting and publish one no-clobber metrics artifact."""

    def __init__(self, plan_path: str, plan_sha256: str, runtime_metrics_path: str) -> None:
        self.plan_path = Path(plan_path).resolve()
        self.launch_contract_sha256 = plan_sha256
        self.runtime_metrics_path = Path(runtime_metrics_path).resolve()
        self.plan: dict[str, Any] | None = None
        self.accounting: RQ1RuntimeAccounting | None = None
        self.attempt_claim: dict[str, Any] | None = None
        self.initialization_report: dict[str, Any] | None = None
        self.last_log_iteration = 0

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001, ANN201, ARG002
        accelerator = kwargs.get("accelerator")
        env = kwargs.get("env")
        if accelerator is None or env is None:
            raise RQ1TrainingError("RQ1 callback requires accelerator and environment")
        if int(accelerator.num_processes) != 1:
            raise RQ1TrainingError("single-5090 RQ1 accounting requires exactly one process")
        plan = load_json_object(self.plan_path)
        validate_rq1_training_plan(plan)
        if plan.get("launch_contract_sha256") != self.launch_contract_sha256:
            raise RQ1TrainingError("callback launch-contract digest differs from the plan")
        if plan.get(PLAN_DIGEST_FIELD) is None:
            raise RQ1TrainingError("callback plan lacks a canonical plan identity")
        expected_metrics = Path(plan["runtime_artifacts"]["runtime_metrics"]).resolve()
        if self.runtime_metrics_path != expected_metrics:
            raise RQ1TrainingError("callback runtime-metrics path differs from the plan")
        if self.runtime_metrics_path.exists():
            raise FileExistsError(
                f"RQ1 runtime metrics already exist; resume/overwrite is forbidden: {self.runtime_metrics_path}"
            )
        receipt_path = Path(plan["runtime_artifacts"]["receipt"]).resolve()
        if receipt_path.exists():
            raise FileExistsError(f"RQ1 receipt already exists; rerun is forbidden: {receipt_path}")
        for name in (
            "final_checkpoint",
            "execution_completion",
            "gpu_isolation_monitor",
        ):
            artifact = Path(plan["runtime_artifacts"][name]).resolve()
            if artifact.exists() or artifact.is_symlink():
                raise FileExistsError(
                    f"prior RQ1 {name} artifact exists; rerun is forbidden: {artifact}"
                )
        if Path(str(args.output_dir)).resolve() != Path(plan["output_dir"]).resolve():
            raise RQ1TrainingError("trainer output directory differs from the RQ1 plan")
        audit_saved_rq1_training_config(plan["runtime_artifacts"]["resolved_config"], plan)
        claim_path = Path(plan["runtime_artifacts"]["attempt_claim"]).resolve()
        claim = load_json_object(claim_path)
        validate_attempt_claim(claim, plan)
        config_path = Path(plan["runtime_artifacts"]["resolved_config"]).resolve()
        if claim_path.stat().st_mtime_ns > config_path.stat().st_mtime_ns:
            raise RQ1TrainingError("attempt claim was not acquired before output creation")
        process_log_path = Path(plan["runtime_artifacts"]["process_log"]).resolve()
        if not process_log_path.is_file() or process_log_path.is_symlink():
            raise RQ1TrainingError("launcher-owned process log was not opened before child startup")
        initialization_report = getattr(env, "_lace_rq1_initialization_report", None)
        if not isinstance(initialization_report, dict):
            raise RQ1TrainingError("strict RQ1 model-only initialization report is absent")
        validate_initialization_report(initialization_report, plan)

        semantics = plan["training_semantics"]
        runtime_schedule = {
            "num_envs_per_rank": int(env.num_envs),
            "world_size": int(accelerator.num_processes),
            "iterations": int(args.num_total_batches),
            "ppo_epochs": int(args.num_ppo_epochs),
            "minibatches_per_epoch": int(args.num_mini_batches),
            "local_minibatch_size": int(args.local_mini_batch_size),
            "per_device_train_batch_size": int(args.per_device_train_batch_size),
            "num_microbatches_per_minibatch": int(args.num_micro_batches),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "decimation": int(env.config.get("decimation")),
            "terrain_type": str(env.config.get("terrain_type")),
        }
        expected_schedule = {key: semantics[key] for key in runtime_schedule}
        if runtime_schedule != expected_schedule:
            raise RQ1TrainingError(
                f"live PPO/environment factorization differs from the plan: "
                f"actual={runtime_schedule}, expected={expected_schedule}"
            )

        motion_lib = getattr(env, "_motion_lib", None)
        motion_command = getattr(env, "motion_command", None)
        if motion_lib is None or motion_command is None:
            raise RQ1TrainingError("RQ1 callback requires the tracking motion command/library")
        assert_fixed_distribution_integrity(motion_lib)
        binding = motion_lib.lace_fixed_distribution_binding
        arm = plan["arm_contract"]
        if binding.get("arm_id") != plan["arm_id"]:
            raise RQ1TrainingError("resident fixed sampler selected a different arm")
        if (
            binding.get(FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD)
            != arm[FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD]
        ):
            raise RQ1TrainingError("resident fixed-sampler binding differs from the plan")
        if list(motion_lib.curr_motion_keys) != arm["motion_keys"]:
            raise RQ1TrainingError("resident motion identity/order differs from the plan")
        if motion_lib.use_adaptive_sampling is not False:
            raise RQ1TrainingError("native adaptive sampling is enabled in an RQ1 run")
        if motion_lib.all_motions_loaded is not True:
            raise RQ1TrainingError("not all D_curriculum motions are resident")
        if bool(getattr(env, "is_evaluating", False)):
            raise RQ1TrainingError("RQ1 training environment entered evaluation mode")
        validate_fixed_distribution_command_modes(
            use_paired_motions=bool(motion_command.cfg.use_paired_motions),
            sample_unique_motions=bool(getattr(motion_command.cfg, "sample_unique_motions", False)),
            is_evaluating=bool(getattr(motion_command, "is_evaluating", False)),
            atlas_probe_mode=bool(getattr(motion_command, "_atlas_probe_enabled", False)),
            multi_object_mode=bool(getattr(motion_command, "_multi_object_mode", False)),
        )
        if getattr(env, "_lace_rq1_accounting", None) is not None:
            raise RQ1TrainingError("RQ1 runtime accounting is already installed")
        accounting = RQ1RuntimeAccounting(plan)
        env._lace_rq1_accounting = accounting  # noqa: SLF001
        self.plan = plan
        self.accounting = accounting
        self.attempt_claim = claim
        self.initialization_report = initialization_report
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001, ANN201, ARG002
        if self.accounting is None:
            raise RQ1TrainingError("RQ1 callback logged before runtime accounting was installed")
        iteration = int(state.global_step)
        if iteration != self.last_log_iteration + 1:
            raise RQ1TrainingError(
                f"RQ1 trainer iterations must be contiguous: expected "
                f"{self.last_log_iteration + 1}, got {iteration}"
            )
        if self.accounting.optimizer_iterations != iteration:
            raise RQ1TrainingError(
                "RQ1 optimizer accounting was not recorded before the trainer log boundary"
            )
        self.last_log_iteration = iteration
        return control

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ANN001, ANN201, ARG002
        env = kwargs.get("env")
        if (
            self.plan is None
            or self.accounting is None
            or self.attempt_claim is None
            or self.initialization_report is None
            or env is None
        ):
            raise RQ1TrainingError("RQ1 callback ended without an initialized accounting contract")
        expected_iterations = int(self.plan["training_semantics"]["iterations"])
        if int(state.global_step) != expected_iterations:
            raise RQ1TrainingError(
                f"RQ1 training ended before the fixed horizon: {state.global_step}/{expected_iterations}"
            )
        if self.last_log_iteration != expected_iterations:
            raise RQ1TrainingError("RQ1 trainer omitted a final contiguous log boundary")
        draw_report = fixed_distribution_draw_report(env._motion_lib)  # noqa: SLF001
        final_checkpoint_path = Path(self.plan["runtime_artifacts"]["final_checkpoint"]).resolve()
        if not final_checkpoint_path.is_file() or final_checkpoint_path.stat().st_size <= 0:
            raise RQ1TrainingError(
                "the final model checkpoint was not saved before RQ1 accounting closed"
            )
        final_checkpoint = {
            "path": str(final_checkpoint_path),
            "bytes": final_checkpoint_path.stat().st_size,
            "sha256": file_sha256(final_checkpoint_path),
        }
        metrics = self.accounting.build_runtime_metrics(
            draw_report,
            final_checkpoint=final_checkpoint,
            attempt_claim=self.attempt_claim,
            initialization_report=self.initialization_report,
        )
        write_new_json(self.runtime_metrics_path, metrics)
        # Verify the just-published bytes before handing control back to the
        # process-level receipt builder.
        published = load_json_object(self.runtime_metrics_path)
        validate_runtime_metrics(published, self.plan)
        if published != metrics or not file_sha256(self.runtime_metrics_path):
            raise RQ1TrainingError("published RQ1 runtime metrics bytes drifted")
        return control
