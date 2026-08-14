# envs/tasks.py

import os
import random
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from typing import Callable


@dataclass
class TaskSpec:
    task_id: str
    difficulty: str  # "easy", "medium", "hard"
    description: str
    setup: Callable[[str], None]  # receives workdir, creates initial state
    verify: Callable[[str], float]  # receives workdir, returns score 0.0-1.0


def _write(workdir: str, name: str, content: str) -> None:
    with open(os.path.join(workdir, name), "w", newline="\n") as f:
        f.write(content)


def _read(workdir: str, name: str) -> str | None:
    path = os.path.join(workdir, name)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return f.read()


def _exists(workdir: str, name: str) -> bool:
    return os.path.exists(os.path.join(workdir, name))


# ---------------------------------------------------------------------------
# Easy tasks
# ---------------------------------------------------------------------------

def _easy_create_hello():
    def setup(wd):
        pass

    def verify(wd):
        content = _read(wd, "hello.txt")
        if content is not None and "hello world" in content.strip().lower():
            return 1.0
        return 0.0

    return TaskSpec(
        task_id="easy_create_hello",
        difficulty="easy",
        description="Create a file named 'hello.txt' containing exactly the text 'hello world'.",
        setup=setup,
        verify=verify,
    )


def _easy_create_dir():
    def setup(wd):
        pass

    def verify(wd):
        return 1.0 if os.path.isdir(os.path.join(wd, "mydir")) else 0.0

    return TaskSpec(
        task_id="easy_create_dir",
        difficulty="easy",
        description="Create a directory called 'mydir'.",
        setup=setup,
        verify=verify,
    )


def _easy_count_lines():
    lines = random.randint(5, 20)
    content = "\n".join(f"line {i}" for i in range(1, lines + 1)) + "\n"

    def setup(wd):
        _write(wd, "data.txt", content)

    def verify(wd):
        result = _read(wd, "count.txt")
        if result is None:
            return 0.0
        try:
            if int(result.strip()) == lines:
                return 1.0
        except ValueError:
            pass
        return 0.0

    return TaskSpec(
        task_id="easy_count_lines",
        difficulty="easy",
        description="Count the number of lines in 'data.txt' and write just the number to 'count.txt'.",
        setup=setup,
        verify=verify,
    )


def _easy_copy_file():
    content = "This is the source file content.\n"

    def setup(wd):
        _write(wd, "source.txt", content)

    def verify(wd):
        result = _read(wd, "dest.txt")
        return 1.0 if result == content else 0.0

    return TaskSpec(
        task_id="easy_copy_file",
        difficulty="easy",
        description="Copy 'source.txt' to 'dest.txt'.",
        setup=setup,
        verify=verify,
    )


def _easy_rename_file():
    content = "rename me\n"

    def setup(wd):
        _write(wd, "old.txt", content)

    def verify(wd):
        if _exists(wd, "old.txt"):
            return 0.0
        result = _read(wd, "new.txt")
        return 1.0 if result == content else 0.0

    return TaskSpec(
        task_id="easy_rename_file",
        difficulty="easy",
        description="Rename 'old.txt' to 'new.txt'.",
        setup=setup,
        verify=verify,
    )


def _easy_append_line():
    original = "first line\nsecond line\n"

    def setup(wd):
        _write(wd, "file.txt", original)

    def verify(wd):
        result = _read(wd, "file.txt")
        if result is None:
            return 0.0
        if result.startswith(original) and "last line" in result[len(original):]:
            return 1.0
        return 0.0

    return TaskSpec(
        task_id="easy_append_line",
        difficulty="easy",
        description="Append the text 'last line' to the end of 'file.txt'.",
        setup=setup,
        verify=verify,
    )


def _easy_head_file():
    lines = [f"line {i}" for i in range(1, 11)]
    content = "\n".join(lines) + "\n"
    expected = "\n".join(lines[:3])

    def setup(wd):
        _write(wd, "data.txt", content)

    def verify(wd):
        result = _read(wd, "top.txt")
        if result is None:
            return 0.0
        if result.strip() == expected.strip():
            return 1.0
        return 0.0

    return TaskSpec(
        task_id="easy_head_file",
        difficulty="easy",
        description="Write the first 3 lines of 'data.txt' to a new file called 'top.txt'.",
        setup=setup,
        verify=verify,
    )


def _easy_create_empty():
    def setup(wd):
        pass

    def verify(wd):
        path = os.path.join(wd, "empty.txt")
        if os.path.isfile(path) and os.path.getsize(path) == 0:
            return 1.0
        return 0.0

    return TaskSpec(
        task_id="easy_create_empty",
        difficulty="easy",
        description="Create an empty file called 'empty.txt'.",
        setup=setup,
        verify=verify,
    )


def _easy_delete_file():
    def setup(wd):
        _write(wd, "trash.txt", "delete me\n")

    def verify(wd):
        return 0.0 if _exists(wd, "trash.txt") else 1.0

    return TaskSpec(
        task_id="easy_delete_file",
        difficulty="easy",
        description="Delete the file 'trash.txt'.",
        setup=setup,
        verify=verify,
    )


def _easy_list_files():
    def setup(wd):
        for name in ["alpha.txt", "beta.txt", "gamma.txt"]:
            _write(wd, name, "x\n")

    def verify(wd):
        result = _read(wd, "listing.txt")
        if result is None:
            return 0.0
        found = set()
        for name in ["alpha.txt", "beta.txt", "gamma.txt"]:
            if name in result:
                found.add(name)
        return 1.0 if len(found) == 3 else len(found) / 3.0

    return TaskSpec(
        task_id="easy_list_files",
        difficulty="easy",
        description="List all files in the current directory and write the output to 'listing.txt'.",
        setup=setup,
        verify=verify,
    )


# ---------------------------------------------------------------------------
# Medium tasks
# ---------------------------------------------------------------------------

def _med_find_txt_files():
    def setup(wd):
        _write(wd, "a.txt", "a\n")
        _write(wd, "b.txt", "b\n")
        _write(wd, "c.log", "c\n")
        os.makedirs(os.path.join(wd, "sub"), exist_ok=True)
        _write(wd, os.path.join("sub", "d.txt"), "d\n")

    def verify(wd):
        result = _read(wd, "results.txt")
        if result is None:
            return 0.0
        needed = {"a.txt", "b.txt", "d.txt"}
        found = sum(1 for name in needed if name in result)
        # should NOT include c.log
        penalty = 0.5 if "c.log" in result else 0.0
        return max(0.0, found / len(needed) - penalty)

    return TaskSpec(
        task_id="med_find_txt",
        difficulty="medium",
        description="Find all '.txt' files (recursively) and write their paths to 'results.txt', one per line.",
        setup=setup,
        verify=verify,
    )


def _med_sort_file():
    words = ["banana", "apple", "cherry", "date", "elderberry"]
    content = "\n".join(words) + "\n"
    expected = "\n".join(sorted(words)) + "\n"

    def setup(wd):
        _write(wd, "unsorted.txt", content)

    def verify(wd):
        result = _read(wd, "sorted.txt")
        if result is None:
            return 0.0
        return 1.0 if result.strip() == expected.strip() else 0.0

    return TaskSpec(
        task_id="med_sort_file",
        difficulty="medium",
        description="Sort 'unsorted.txt' alphabetically and save the result to 'sorted.txt'.",
        setup=setup,
        verify=verify,
    )


def _med_grep_errors():
    log_lines = [
        "2024-01-01 INFO started",
        "2024-01-01 ERROR disk full",
        "2024-01-02 INFO running",
        "2024-01-02 ERROR timeout",
        "2024-01-03 INFO stopped",
    ]
    content = "\n".join(log_lines) + "\n"

    def setup(wd):
        _write(wd, "app.log", content)

    def verify(wd):
        result = _read(wd, "errors.txt")
        if result is None:
            return 0.0
        lines = [l.strip() for l in result.strip().splitlines() if l.strip()]
        expected = [l for l in log_lines if "ERROR" in l]
        if lines == expected:
            return 1.0
        # partial credit
        hits = sum(1 for l in expected if l in lines)
        return hits / len(expected) * 0.8

    return TaskSpec(
        task_id="med_grep_errors",
        difficulty="medium",
        description="Extract all lines containing 'ERROR' from 'app.log' and write them to 'errors.txt'.",
        setup=setup,
        verify=verify,
    )


def _med_count_unique_words():
    words = ["hello", "world", "hello", "foo", "bar", "world", "baz", "foo"]
    content = " ".join(words) + "\n"
    expected = len(set(words))

    def setup(wd):
        _write(wd, "words.txt", content)

    def verify(wd):
        result = _read(wd, "count.txt")
        if result is None:
            return 0.0
        try:
            return 1.0 if int(result.strip()) == expected else 0.0
        except ValueError:
            return 0.0

    return TaskSpec(
        task_id="med_unique_words",
        difficulty="medium",
        description="Count the number of unique words in 'words.txt' and write just the number to 'count.txt'.",
        setup=setup,
        verify=verify,
    )


def _med_replace_text():
    content = "foo is great\nfoo and bar\nno match here\n"
    expected = content.replace("foo", "bar")

    def setup(wd):
        _write(wd, "input.txt", content)

    def verify(wd):
        result = _read(wd, "input.txt")
        if result is None:
            return 0.0
        return 1.0 if result == expected else 0.0

    return TaskSpec(
        task_id="med_replace_text",
        difficulty="medium",
        description="Replace all occurrences of 'foo' with 'bar' in 'input.txt' (modify the file in-place).",
        setup=setup,
        verify=verify,
    )


def _med_merge_files():
    content_a = "line A1\nline A2\n"
    content_b = "line B1\nline B2\n"

    def setup(wd):
        _write(wd, "a.txt", content_a)
        _write(wd, "b.txt", content_b)

    def verify(wd):
        result = _read(wd, "merged.txt")
        if result is None:
            return 0.0
        if content_a in result and content_b in result:
            return 1.0
        return 0.0

    return TaskSpec(
        task_id="med_merge_files",
        difficulty="medium",
        description="Merge 'a.txt' and 'b.txt' into a single file called 'merged.txt' (a.txt content first, then b.txt).",
        setup=setup,
        verify=verify,
    )


def _med_deep_dir():
    def setup(wd):
        pass

    def verify(wd):
        path = os.path.join(wd, "a", "b", "c", "deep.txt")
        if os.path.isfile(path):
            return 1.0
        # partial: directory exists but no file
        if os.path.isdir(os.path.join(wd, "a", "b", "c")):
            return 0.5
        return 0.0

    return TaskSpec(
        task_id="med_deep_dir",
        difficulty="medium",
        description="Create the directory structure 'a/b/c/' and place a file called 'deep.txt' inside the deepest directory.",
        setup=setup,
        verify=verify,
    )


def _med_csv_column():
    csv_content = "name,age,city\nalice,30,london\nbob,25,paris\ncharlie,35,tokyo\n"

    def setup(wd):
        _write(wd, "data.csv", csv_content)

    def verify(wd):
        result = _read(wd, "names.txt")
        if result is None:
            return 0.0
        lines = [l.strip() for l in result.strip().splitlines() if l.strip()]
        expected = ["name", "alice", "bob", "charlie"]
        # accept with or without header
        if lines == expected or lines == expected[1:]:
            return 1.0
        return 0.0

    return TaskSpec(
        task_id="med_csv_column",
        difficulty="medium",
        description="Extract the first column (name) from 'data.csv' and write it to 'names.txt', one name per line.",
        setup=setup,
        verify=verify,
    )


def _med_tail_file():
    lines = [f"entry {i}" for i in range(1, 21)]
    content = "\n".join(lines) + "\n"
    expected = "\n".join(lines[-5:])

    def setup(wd):
        _write(wd, "log.txt", content)

    def verify(wd):
        result = _read(wd, "last5.txt")
        if result is None:
            return 0.0
        return 1.0 if result.strip() == expected.strip() else 0.0

    return TaskSpec(
        task_id="med_tail_file",
        difficulty="medium",
        description="Write the last 5 lines of 'log.txt' to a file called 'last5.txt'.",
        setup=setup,
        verify=verify,
    )


def _med_word_freq():
    content = "apple banana apple cherry banana apple\n"

    def setup(wd):
        _write(wd, "words.txt", content)

    def verify(wd):
        result = _read(wd, "freq.txt")
        if result is None:
            return 0.0
        # Parse 'count word' (or 'word count') pairs so counts must actually
        # be associated with the right word, not just appear anywhere.
        expected = {"apple": 3, "banana": 2, "cherry": 1}
        counts = {}
        ordered_counts = []
        for line in result.strip().splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            a, b = parts
            if a.isdigit():
                cnt, word = int(a), b
            elif b.isdigit():
                word, cnt = a, int(b)
            else:
                continue
            if word in expected and word not in counts:
                counts[word] = cnt
                ordered_counts.append(cnt)
        correct = sum(1 for w, c in expected.items() if counts.get(w) == c)
        if correct == 3 and ordered_counts == sorted(ordered_counts, reverse=True):
            return 1.0
        # partial: right counts but wrong order / missing entries
        return correct / 3.0 * 0.8

    return TaskSpec(
        task_id="med_word_freq",
        difficulty="medium",
        description="Count the frequency of each word in 'words.txt' and write the results to 'freq.txt' (format: 'count word', one per line, sorted by frequency descending).",
        setup=setup,
        verify=verify,
    )


# ---------------------------------------------------------------------------
# Hard tasks
# ---------------------------------------------------------------------------

def _hard_log_pipeline():
    timestamps = [
        "2024-01-03T10:00:00",
        "2024-01-01T08:00:00",
        "2024-01-02T09:00:00",
        "2024-01-01T08:00:00",  # duplicate
        "2024-01-03T10:00:00",  # duplicate
        "2024-01-04T11:00:00",
    ]
    log_lines = [f"{ts} INFO event" for ts in timestamps]
    content = "\n".join(log_lines) + "\n"
    expected_ts = sorted(set(timestamps))

    def setup(wd):
        _write(wd, "server.log", content)

    def verify(wd):
        result = _read(wd, "timestamps.txt")
        if result is None:
            return 0.0
        lines = [l.strip() for l in result.strip().splitlines() if l.strip()]
        if lines == expected_ts:
            return 1.0
        # partial: right count, right elements
        if set(lines) == set(expected_ts):
            return 0.7
        return 0.0

    return TaskSpec(
        task_id="hard_log_pipeline",
        difficulty="hard",
        description=(
            "Process 'server.log': extract just the timestamps (first field of each line), "
            "remove duplicates, sort them chronologically, and save to 'timestamps.txt' (one per line)."
        ),
        setup=setup,
        verify=verify,
    )


def _hard_shell_script():
    def setup(wd):
        pass

    def verify(wd):
        script_path = os.path.join(wd, "greet.sh")
        if not os.path.isfile(script_path):
            return 0.0
        score = 0.0
        # check it's executable
        if os.access(script_path, os.X_OK):
            score += 0.1
        # actually run the script: it must print 'hello' to stdout
        # (defeats `echo hello > greet.sh`, which is not a working script)
        bash = shutil.which("bash") or "bash"
        try:
            proc = subprocess.run(
                [bash, script_path],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=wd,
            )
            if "hello" in proc.stdout.lower():
                score += 0.4
        except (subprocess.TimeoutExpired, OSError):
            pass
        # check output.txt has "hello"
        output = _read(wd, "output.txt")
        if output and "hello" in output.lower():
            score += 0.5
        return min(score, 1.0)

    return TaskSpec(
        task_id="hard_shell_script",
        difficulty="hard",
        description=(
            "Create a shell script called 'greet.sh' that prints 'hello' to stdout. "
            "Make it executable, run it, and redirect its output to 'output.txt'."
        ),
        setup=setup,
        verify=verify,
    )


def _hard_find_large_files():
    def setup(wd):
        # create files of various sizes
        _write(wd, "small.txt", "x" * 50 + "\n")
        _write(wd, "medium.txt", "x" * 200 + "\n")
        _write(wd, "big.txt", "x" * 500 + "\n")
        os.makedirs(os.path.join(wd, "subdir"), exist_ok=True)
        _write(wd, os.path.join("subdir", "huge.txt"), "x" * 1000 + "\n")

    def verify(wd):
        result = _read(wd, "large.txt")
        if result is None:
            return 0.0
        # files > 100 bytes: medium.txt, big.txt, huge.txt
        needed = ["huge.txt", "big.txt", "medium.txt"]  # largest first
        found = sum(1 for name in needed if name in result)
        # should NOT include small.txt
        penalty = 0.3 if "small.txt" in result else 0.0
        base = max(0.0, found / len(needed) - penalty)
        if found == len(needed) and penalty == 0.0:
            # full credit only if sorted largest-first, as the task requires
            positions = [result.find(name) for name in needed]
            return 1.0 if positions == sorted(positions) else 0.8
        return base

    return TaskSpec(
        task_id="hard_find_large",
        difficulty="hard",
        description=(
            "Find all files larger than 100 bytes (recursively) and write their paths to 'large.txt', "
            "sorted by size (largest first), one per line."
        ),
        setup=setup,
        verify=verify,
    )


def _hard_tar_archive():
    def setup(wd):
        os.makedirs(os.path.join(wd, "project"), exist_ok=True)
        _write(wd, os.path.join("project", "main.py"), "print('hello')\n")
        _write(wd, os.path.join("project", "readme.md"), "# Project\n")

    def verify(wd):
        # the archive must be a real gzipped tar containing the project files
        # (defeats `echo junk > archive.tar.gz`)
        for name in ["archive.tar.gz", "archive.tgz"]:
            path = os.path.join(wd, name)
            if not os.path.isfile(path):
                continue
            try:
                with tarfile.open(path, "r:gz") as tf:
                    members = tf.getnames()
            except (tarfile.TarError, OSError, EOFError):
                return 0.0
            has_main = any(m.endswith("main.py") for m in members)
            has_readme = any(m.endswith("readme.md") for m in members)
            if has_main and has_readme:
                return 1.0
            # valid archive but missing the project files
            return 0.3 if members else 0.0
        return 0.0

    return TaskSpec(
        task_id="hard_tar_archive",
        difficulty="hard",
        description="Create a tar.gz archive called 'archive.tar.gz' containing the 'project/' directory and all its files.",
        setup=setup,
        verify=verify,
    )


def _hard_json_extract():
    json_content = """{
  "server": {
    "host": "localhost",
    "port": 8080
  },
  "database": {
    "name": "mydb",
    "port": 5432
  }
}
"""

    def setup(wd):
        _write(wd, "config.json", json_content)

    def verify(wd):
        result = _read(wd, "dbname.txt")
        if result is None:
            return 0.0
        return 1.0 if result.strip() == "mydb" else 0.0

    return TaskSpec(
        task_id="hard_json_extract",
        difficulty="hard",
        description=(
            "Extract the database name from 'config.json' (the value of .database.name) "
            "and write it to 'dbname.txt'."
        ),
        setup=setup,
        verify=verify,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def get_all_tasks() -> list[TaskSpec]:
    return [
        # Easy
        _easy_create_hello(),
        _easy_create_dir(),
        _easy_count_lines(),
        _easy_copy_file(),
        _easy_rename_file(),
        _easy_append_line(),
        _easy_head_file(),
        _easy_create_empty(),
        _easy_delete_file(),
        _easy_list_files(),
        # Medium
        _med_find_txt_files(),
        _med_sort_file(),
        _med_grep_errors(),
        _med_count_unique_words(),
        _med_replace_text(),
        _med_merge_files(),
        _med_deep_dir(),
        _med_csv_column(),
        _med_tail_file(),
        _med_word_freq(),
        # Hard
        _hard_log_pipeline(),
        _hard_shell_script(),
        _hard_find_large_files(),
        _hard_tar_archive(),
        _hard_json_extract(),
    ]
