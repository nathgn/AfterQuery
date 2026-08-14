# tests/test_tasks.py
#
# Verifier sanity tests. Runs with pytest or directly:
#   python tests/test_tasks.py
#
# Three guarantees:
#   1. Every task is solvable: a reference single-command solution scores 1.0.
#   2. Doing nothing scores 0.0 on every task.
#   3. Known reward-hacking commands do NOT score 1.0.
#   4. Partial credit flows through the training reward function.

import sys

sys.path.insert(0, ".")

from envs.rewards import make_reward_func
from envs.tasks import get_all_tasks
from envs.terminalbench_client import TerminalBenchClient
from envs.terminalbench_env import TerminalBenchEnv

# One known-good single command per task (what a competent model should emit).
REFERENCE_SOLUTIONS = {
    # Easy
    "easy_create_hello": "echo 'hello world' > hello.txt",
    "easy_create_dir": "mkdir mydir",
    "easy_count_lines": "wc -l < data.txt | tr -d ' ' > count.txt",
    "easy_copy_file": "cp source.txt dest.txt",
    "easy_rename_file": "mv old.txt new.txt",
    "easy_append_line": "echo 'last line' >> file.txt",
    "easy_head_file": "head -n 3 data.txt > top.txt",
    "easy_create_empty": "touch empty.txt",
    "easy_delete_file": "rm trash.txt",
    "easy_list_files": "ls > listing.txt",
    # Medium
    "med_find_txt": "find . -name '*.txt' -not -name 'results.txt' > results.txt",
    "med_sort_file": "sort unsorted.txt > sorted.txt",
    "med_grep_errors": "grep ERROR app.log > errors.txt",
    "med_unique_words": "tr ' ' '\\n' < words.txt | grep -v '^$' | sort -u | wc -l | tr -d ' ' > count.txt",
    "med_replace_text": "sed -i.bak 's/foo/bar/g' input.txt",
    "med_merge_files": "cat a.txt b.txt > merged.txt",
    "med_deep_dir": "mkdir -p a/b/c && touch a/b/c/deep.txt",
    "med_csv_column": "cut -d, -f1 data.csv > names.txt",
    "med_tail_file": "tail -n 5 log.txt > last5.txt",
    "med_word_freq": "tr ' ' '\\n' < words.txt | grep -v '^$' | sort | uniq -c | sort -rn | awk '{print $1, $2}' > freq.txt",
    # Hard
    "hard_log_pipeline": "awk '{print $1}' server.log | sort -u > timestamps.txt",
    "hard_shell_script": "printf '#!/bin/bash\\necho hello\\n' > greet.sh && chmod +x greet.sh && ./greet.sh > output.txt",
    "hard_find_large": "find . -type f -size +100c -exec ls -S {} + > large.txt",
    "hard_tar_archive": "tar -czf archive.tar.gz project/",
    "hard_json_extract": "grep -A2 '\"database\"' config.json | grep '\"name\"' | sed 's/.*: *\"\\(.*\\)\".*/\\1/' > dbname.txt",
}

# Commands that used to score 1.0 without doing the task. Each must now
# score strictly below 1.0.
REWARD_HACKS = {
    "hard_shell_script": "echo hello > greet.sh && echo hello > output.txt",
    "hard_tar_archive": "echo junk > archive.tar.gz",
    "med_word_freq": "echo 'apple banana cherry 1 2 3' > freq.txt",
    "hard_find_large": "printf 'medium.txt\\nbig.txt\\nhuge.txt\\n' > large.txt",  # unsorted
}


def _run_command(spec, command: str) -> float:
    client = TerminalBenchClient(task_specs=[spec])
    env = TerminalBenchEnv(client, max_steps=1)
    env.reset()
    _obs, _reward, _done, info = env.step(command)
    return info["success_score"]


def test_all_tasks_have_reference_solutions():
    task_ids = {t.task_id for t in get_all_tasks()}
    assert task_ids == set(REFERENCE_SOLUTIONS), (
        f"missing: {task_ids - set(REFERENCE_SOLUTIONS)}, "
        f"stale: {set(REFERENCE_SOLUTIONS) - task_ids}"
    )


def test_reference_solutions_score_full():
    failures = []
    for spec in get_all_tasks():
        score = _run_command(spec, REFERENCE_SOLUTIONS[spec.task_id])
        if score < 1.0:
            failures.append(f"{spec.task_id}: {score:.2f}")
    assert not failures, f"reference solutions below 1.0: {failures}"


def test_noop_scores_zero():
    failures = []
    for spec in get_all_tasks():
        score = _run_command(spec, "echo noop")
        if score != 0.0:
            failures.append(f"{spec.task_id}: {score:.2f}")
    assert not failures, f"noop scored above 0.0: {failures}"


def test_reward_hacks_do_not_score_full():
    specs = {t.task_id: t for t in get_all_tasks()}
    failures = []
    for task_id, cheat in REWARD_HACKS.items():
        score = _run_command(specs[task_id], cheat)
        if score >= 1.0:
            failures.append(f"{task_id}: cheat scored {score:.2f}")
    assert not failures, f"reward hacks still pay: {failures}"


def test_reward_func_matches_task_and_grants_partial_credit():
    reward_func = make_reward_func()
    specs = {t.task_id: t for t in get_all_tasks()}

    def prompt_for(task_id):
        return f"[terminalbench]\nTask: {specs[task_id].description}\nCommand:"

    prompts = [
        prompt_for("easy_create_hello"),
        prompt_for("med_deep_dir"),
        prompt_for("easy_create_dir"),
    ]
    completions = [
        "echo 'hello world' > hello.txt",  # full credit
        "mkdir -p a/b/c",                  # partial credit (0.5: dirs but no file)
        "echo 'hello world' > hello.txt",  # right command, WRONG task -> low
    ]
    rewards = reward_func(prompts, completions)

    assert rewards[0] > 0.9, f"full solution got {rewards[0]}"
    assert 0.3 < rewards[1] < 0.7, f"partial credit not granted: {rewards[1]}"
    assert rewards[2] < 0.1, f"mismatched task scored {rewards[2]}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    sys.exit(1 if failed else 0)
