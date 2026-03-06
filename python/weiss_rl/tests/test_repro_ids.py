from __future__ import annotations

from pathlib import Path

from weiss_rl.repro import derive_actor_seed, derive_episode_seed, parse_seed_file


def test_seed_derivation_is_deterministic() -> None:
    actor_seed_a = derive_actor_seed(20260212, actor_id=4)
    actor_seed_b = derive_actor_seed(20260212, actor_id=4)
    assert actor_seed_a == actor_seed_b

    episode_seed_a = derive_episode_seed(actor_seed_a, env_id=1, episode_index=77)
    episode_seed_b = derive_episode_seed(actor_seed_b, env_id=1, episode_index=77)
    assert episode_seed_a == episode_seed_b


def test_parse_seed_file_valid(tmp_path: Path) -> None:
    seed_file = tmp_path / "seeds.txt"
    seed_file.write_text("123\n456\n789\n")
    seeds = parse_seed_file(seed_file)
    assert seeds == [123, 456, 789]


def test_parse_seed_file_rejects_comments_and_blanks(tmp_path: Path) -> None:
    seed_file = tmp_path / "seeds.txt"
    seed_file.write_text("123\n# comment\n\n456\n")
    seeds = parse_seed_file(seed_file)
    assert seeds == [123, 456]


def test_parse_seed_file_invalid_line(tmp_path: Path) -> None:
    seed_file = tmp_path / "seeds.txt"
    seed_file.write_text("123\nabc\n456\n")
    try:
        parse_seed_file(seed_file)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid seed on line 2" in str(e)


def test_parse_seed_file_out_of_range(tmp_path: Path) -> None:
    seed_file = tmp_path / "seeds.txt"
    seed_file.write_text("123\n-1\n456\n")
    try:
        parse_seed_file(seed_file)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "out of u64 range" in str(e)
