"""
RLSD 训练主入口。

用法：
    bash recipe/RLSD/run_rlsd.sh
    python recipe/RLSD/main_rlsd.py [hydra overrides]
"""

import os
import socket

# 必须在 import torch/verl 之前设置
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
if "/usr/local/cuda/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")

# Python tempfile / Torch 等与 Ray spill 默认写根分区 /tmp；改到大容量盘
_verl_tmp = os.path.abspath(os.path.expanduser(os.environ.get("VERL_TMP_ROOT", "/data3/yyy/tmp")))
os.makedirs(_verl_tmp, exist_ok=True)
_ray_tmp = os.path.join(_verl_tmp, "ray")
os.makedirs(_ray_tmp, exist_ok=True)
os.environ.setdefault("TMPDIR", _verl_tmp)
os.environ.setdefault("RAY_TMPDIR", _ray_tmp)

import hydra
import ray
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="rlsd_trainer", version_base=None)
def main(config):
    run_rlsd(config)


def run_rlsd(config) -> None:
    if not ray.is_initialized():
        ray.init(
            runtime_env={"env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARNING",
                "TORCH_COMPILE_DISABLE": "1",
                "CUDA_HOME": "/usr/local/cuda",
                "TMPDIR": os.environ["TMPDIR"],
                "RAY_TMPDIR": os.environ["RAY_TMPDIR"],
            }},
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local
        from verl.utils import hf_tokenizer
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.utils.dataset.rl_dataset import collate_fn

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # ── 模型路径 & Tokenizer ──────────────────────────────────────
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=config.data.get("trust_remote_code", False))

        # ── Worker 类（MRSD 自定义 worker，替换内部 actor 的 loss）────
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from recipe.RLSD.rlsd.rlsd_worker import MRSDActorRolloutWorker
        from recipe.RLSD.rlsd.rlsd_trainer import MRSDTrainer
        from recipe.RLSD.rlsd.dataset import MRSDDataset

        actor_rollout_cls = MRSDActorRolloutWorker
        ray_worker_group_cls = RayWorkerGroup

        # ── Role → Worker 映射 ──
        # ActorRollout: 承载 actor + rollout + ref（colocated）
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {Role.ActorRollout: global_pool_id}

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=mapping,
        )

        # ── 标准 verl 数据集（只用于满足父类 / dataloader 初始化；MRSD fit 不迭代 train_dataloader）
        from omegaconf import ListConfig
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        def _data_files_to_list(fs) -> list[str]:
            if fs is None:
                return []
            if isinstance(fs, (list, tuple, ListConfig)):
                return [str(x) for x in fs]
            return [str(fs)]

        def _parquet_has_column(path: str, col: str) -> bool:
            import pyarrow.parquet as pq

            return col in pq.read_schema(path).names

        train_paths = _data_files_to_list(config.data.train_files)
        if not train_paths:
            raise ValueError("config.data.train_files 不能为空")
        if len(train_paths) > 1:
            print("[main] WARN: MRSDDataset.from_parquet 仅使用 train_files 列表中的第一个路径")
        train_pool_src = train_paths[0]

        rlhf_train_spec = config.data.train_files
        if _parquet_has_column(train_paths[0], "problem") and not _parquet_has_column(
            train_paths[0], "prompt"
        ):
            val_paths = _data_files_to_list(config.data.val_files)
            if not val_paths:
                raise ValueError(
                    "训练集为 OpenThoughts 布局（有 problem、无 prompt）时，必须为 data.val_files "
                    "提供至少一个含 prompt 列的 verl parquet，以供 RLHF Dataset 占位初始化"
                )
            rlhf_train_spec = val_paths[0]
            print(
                f"[main] 训练题池使用 OpenThoughts parquet；RLHF train_dataset 占位文件: {rlhf_train_spec}"
            )

        train_dataset = create_rl_dataset(rlhf_train_spec, config.data, tokenizer, None)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, None)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # ── MRSD 问题数据集 ───────────────────────────────────────────
        mrsd_cfg = config.mrsd
        mrsd_problems_path = config.data.get("mrsd_problems_path", None)
        dataset_kwargs = {}
        ds_seed = OmegaConf.select(mrsd_cfg, "dataset_seed", default=None)
        if ds_seed is not None:
            dataset_kwargs["seed"] = int(ds_seed)
        if mrsd_problems_path:
            print(f"[main] 从 jsonl 加载训练题池: {mrsd_problems_path}")
            mrsd_dataset = MRSDDataset.from_pass_at_k_results(
                pass_at_k_jsonl=mrsd_problems_path,
                type_b_only=True,
                **dataset_kwargs,
            )
        else:
            print(f"[main] mrsd_problems_path 未设置，从 train parquet 加载题池: {train_pool_src}")
            mrsd_dataset = MRSDDataset.from_parquet(train_pool_src, **dataset_kwargs)
        print(f"[main] 共 {len(mrsd_dataset)} 道题目")

        # ── 训练器 ────────────────────────────────────────────────────
        trainer = MRSDTrainer(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            mrsd_dataset=mrsd_dataset,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
