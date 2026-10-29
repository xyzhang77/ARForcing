import argparse
import os
import shutil
import time
from omegaconf import OmegaConf

from trainer import DiffusionTrainer, GANTrainer, ODETrainer, ScoreDistillationTrainer

CODE_SUFFIXES = (".py", ".yaml", ".yml", ".sh", ".txt")
SKIP_DIRS = {"__pycache__", "logs", "wandb", "videos", "images", "vbench"}


def backup_run(config_path, config, logdir):
    backup_dir = os.path.join(logdir, "backup")
    os.makedirs(backup_dir, exist_ok=True)
    shutil.copy2(config_path, os.path.join(backup_dir, "config_original.yaml"))
    OmegaConf.save(config, os.path.join(backup_dir, "config_merged.yaml"))

    root = os.path.dirname(os.path.abspath(__file__))
    logdir_abs = os.path.abspath(logdir)
    for cur, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS
                   and os.path.abspath(os.path.join(cur, d)) != logdir_abs]
        for f in files:
            if f.endswith(CODE_SUFFIXES):
                dst = os.path.join(backup_dir, "code", os.path.relpath(os.path.join(cur, f), root))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(os.path.join(cur, f), dst)
    print(f"Config and code backed up to {backup_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Experiment directory, used as-is (default: logs/<config.exp_name>_<timestamp>)")
    parser.add_argument("--tensorboard-save-dir", type=str, default="", help="TensorBoard log directory (default: <logdir>/tensorboard)")
    parser.add_argument("--disable-tensorboard", action="store_true")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    # get the filename of config_path
    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    # layout: <logdir>/{checkpoints,tensorboard,backup}; only rank 0 writes there
    exp_name = f"{config.get('exp_name') or config_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    config.logdir = args.logdir or os.path.join("logs", exp_name)
    config.checkpoint_dir = os.path.join(config.logdir, "checkpoints")
    config.tensorboard_save_dir = args.tensorboard_save_dir or os.path.join(config.logdir, "tensorboard")
    config.disable_tensorboard = args.disable_tensorboard

    # torchrun sets RANK; dist isn't initialized yet here
    if int(os.environ.get("RANK", 0)) == 0:
        backup_run(args.config_path, config, config.logdir)

    if config.trainer == "diffusion":
        trainer = DiffusionTrainer(config)
    elif config.trainer == "gan":
        trainer = GANTrainer(config)
    elif config.trainer == "ode":
        trainer = ODETrainer(config)
    elif config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config)
    try:
        trainer.train()
    finally:
        if trainer.writer is not None:
            trainer.writer.close()


if __name__ == "__main__":
    main()
