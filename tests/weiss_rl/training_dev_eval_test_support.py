from __future__ import annotations

from types import SimpleNamespace


def make_dev_eval_stack(tmp_path, *, seed_file_name: str = "dev_seeds.txt", required_pairs: int = 2):
    seed_file = tmp_path / seed_file_name
    seed_file.write_text("11\n22\n33\n", encoding="utf-8")
    return SimpleNamespace(
        root=tmp_path,
        seed_sets={"dev_eval": seed_file},
        config=SimpleNamespace(
            evaluation=SimpleNamespace(
                eval_sampling_algorithm="pinned_cdf_pcg_v1",
                eval_device="cpu",
                eval_inference_mode=True,
                seat_swap=True,
                model_sampling_temperature=1.0,
                seed_files={"dev_eval": seed_file.name},
                periodic_dev_eval_paired_seeds=required_pairs,
                periodic_dev_eval_interval_updates=20,
            ),
            reproducibility=SimpleNamespace(seed_files={"dev_eval": seed_file.name}),
        ),
    )
