"""Train a bot for Boost Arena.

    python -m boost_arena.starter.train --run my_first_bot

Stop with Ctrl+C at any time; training resumes from the newest checkpoint when the same
run name is given again. Export a submission with `python -m boost_arena.starter.export`.
"""

import argparse
import multiprocessing
import os
import sys
import time

RUNS_FOLDER = "runs"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a bot for Boost Arena")
    parser.add_argument("--run", default="bot", help="Name of this run. Checkpoints go under runs/<name>/")
    parser.add_argument("--processes", type=int, default=max(2, multiprocessing.cpu_count() - 3), help="Environment processes")
    parser.add_argument("--device", default="cpu", help="Where the network learns: cpu, cuda or mps")
    parser.add_argument("--timesteps", type=int, default=1_300_000_000, help="Stop after this many timesteps")
    parser.add_argument("--layers", default="256,256,256", help="Hidden layer sizes of the policy")
    parser.add_argument("--no-layer-norm", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--entropy", type=float, default=0.02, help="Entropy coefficient: higher explores more")
    parser.add_argument("--steps-per-update", type=int, default=50_000)
    parser.add_argument("--phase-scale", type=float, default=1.0, help="Shrink every curriculum phase start, for quick tests")
    parser.add_argument("--checkpoint-every", type=int, default=5_000_000, help="Timesteps between checkpoints")
    args = parser.parse_args(argv)

    layer_sizes = tuple(int(size) for size in args.layers.split(","))
    run_folder = os.path.abspath(os.path.join(RUNS_FOLDER, args.run))
    os.makedirs(run_folder, exist_ok=True)

    # The environment processes read the trainer's progress from this file to follow the curriculum
    from .curriculum import PROGRESS_FILE, PROGRESS_FILE_VARIABLE, phase_for, write_progress

    progress_path = os.path.join(run_folder, PROGRESS_FILE)
    if not os.path.exists(progress_path):
        write_progress(progress_path, 0, args.phase_scale)
    os.environ[PROGRESS_FILE_VARIABLE] = progress_path

    import numpy as np
    from rlgym_learn import BaseConfigModel, LearningCoordinator, LearningCoordinatorConfigModel, ProcessConfigModel, SerdeTypesModel
    from rlgym_learn.pyany_serde import PyAnySerdeType
    from rlgym_learn_algos.ppo import (
        ExperienceBufferConfigModel, GAETrajectoryProcessor, GAETrajectoryProcessorConfigModel, NumpyExperienceBuffer,
        PPOAgentController, PPOAgentControllerConfigModel, PPOLearnerConfigModel, PPOMetricsLogger,
    )
    from torch.optim import Adam

    from .actor import make_actor_critic
    from .env import build_env

    class ProgressLogger(PPOMetricsLogger):
        """The usual training printout, plus the progress file and a phase announcement."""

        _announced = None

        def collect_agent_metrics(self, data):
            super().collect_agent_metrics(data)
            timesteps = int(data.cumulative_timesteps)
            write_progress(progress_path, timesteps, args.phase_scale)
            phase = phase_for(timesteps, args.phase_scale)
            if phase.name != ProgressLogger._announced:
                ProgressLogger._announced = phase.name
                print(f"\n=== Curriculum phase {phase.name} (from {timesteps:,} timesteps) ===\n", flush=True)

    def optimizers_factory(actor_critic, optimizer_kwargs, agent_controller):
        return [Adam(actor_critic.actor.parameters(), **optimizer_kwargs["actor"]),
                Adam(actor_critic.critic.parameters(), **optimizer_kwargs["critic"])]

    # Resume from the newest checkpoint of this run, if there is one
    from .export import newest_checkpoint

    checkpoint = newest_checkpoint(run_folder)
    if checkpoint:
        print(f"Resuming from {checkpoint}")

    config = LearningCoordinatorConfigModel(
        base_config=BaseConfigModel(
            serde_types=SerdeTypesModel(
                agent_id_serde_type=PyAnySerdeType.STRING(),
                action_serde_type=PyAnySerdeType.NUMPY(np.int64),
                obs_serde_type=PyAnySerdeType.NUMPY(np.float32),
                reward_serde_type=PyAnySerdeType.FLOAT(),
                obs_space_serde_type=PyAnySerdeType.TUPLE((PyAnySerdeType.STRING(), PyAnySerdeType.INT())),
                action_space_serde_type=PyAnySerdeType.TUPLE((PyAnySerdeType.STRING(), PyAnySerdeType.INT())),
            ),
            timestep_limit=args.timesteps,
        ),
        process_config=ProcessConfigModel(n_proc=args.processes),
        agent_controller_config=PPOAgentControllerConfigModel(
            timesteps_per_iteration=args.steps_per_update,
            save_every_ts=args.checkpoint_every,
            checkpoint_load_folder=checkpoint,
            run_name=args.run,
            run_suffix="",
            learner_config=PPOLearnerConfigModel(
                ent_coef=args.entropy,
                batch_size=args.steps_per_update,
                n_minibatches=1,
                n_epochs=1,
                optimizer_named_parameter_group_kwargs={"actor": {"lr": args.learning_rate}, "critic": {"lr": args.learning_rate}},
                device=args.device,
            ),
            experience_buffer_config=ExperienceBufferConfigModel(
                max_size=2 * args.steps_per_update,
                trajectory_processor_config=GAETrajectoryProcessorConfigModel(gamma=phase_for(0, args.phase_scale).gamma),
                device="cpu",
            ),
        ),
        agent_controller_save_folder=os.path.join(run_folder, "checkpoints"),
    )

    coordinator = LearningCoordinator(
        build_env,
        agent_controller=PPOAgentController(
            actor_critic_factory=make_actor_critic(layer_sizes, not args.no_layer_norm, layer_sizes),
            optimizers_factory=optimizers_factory,
            experience_buffer=NumpyExperienceBuffer(GAETrajectoryProcessor()),
            metrics_logger=ProgressLogger(),
            obs_standardizer=None,
        ),
        config=config,
    )
    print(f"Run folder: {run_folder}")
    print(f"Policy layers {layer_sizes}, {args.processes} processes, learning on {args.device}")
    started = time.time()
    coordinator.start()
    print(f"Finished after {(time.time() - started) / 3600:.1f} hours")


if __name__ == "__main__":
    sys.exit(main())
