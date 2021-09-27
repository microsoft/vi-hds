# -------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# -------------------------------------
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).parent.parent

# The text of the copyright notice, without comment characters, newlines or lines of hyphens.
NOTICE = [
    "Copyright (c) Microsoft Corporation.",
    "Licensed under the MIT license.",
]


def construct_full_notice():
    """
    Returns the lines to be added to a file requiring a copyright notice.
    """
    max_len = max(len(line) for line in NOTICE)
    hyphen_line = "# " + ("-" * max_len) + "\n"
    return [hyphen_line] + [f"# {line}\n" for line in NOTICE] + [hyphen_line]


# Lines to actually be added as a copyright notice.
FULL_NOTICE = construct_full_notice()


def check_copyright_notices() -> bool:  # pragma: no cover
    """
    Checks all .py files under [A-Za-z]*, and adds a copyright notice
    where none is found. Returns whether a notice has been added to any file.
    """
    py_files = sorted(ROOT.glob("[A-Za-z]*/**/*.py"))
    return check_copyright_notices_on_files(py_files)


def check_copyright_notices_on_files(py_files: List[Path]) -> bool:
    """
    Checks all the provided files, excluding Emukit, and adds a copyright notice
    where none is found. Returns whether a notice has been added to any file.
    """
    added = False
    for path in py_files:
        if "Emukit" in path.parts:
            continue  # pragma: no cover
        if not has_copyright_notice_or_is_empty(path):
            add_copyright_notice(path)
            added = True
            sys.stderr.write(f"Added copyright notice to: {path}\n")
    return added


def has_copyright_notice_or_is_empty(path: Path) -> bool:
    """
    Returns whether the file already has a copyright notice, or is empty so does not require one.
    """
    to_find = NOTICE
    is_empty = True
    for line in path.open():
        if line.startswith("#"):
            if line.find(to_find[0]) > 0:
                to_find = to_find[1:]
                if not to_find:
                    return True
        else:
            return False
        is_empty = False
    return is_empty


def add_copyright_notice(path: Path) -> None:
    """
    Adds a copyright notice to the file.
    """
    result = []
    added = False
    for line in path.open():
        if not (added or line.startswith("#")):
            result += FULL_NOTICE
            added = True
        result.append(line)
    # This will be needed if the file consists entirely of "#" lines:
    if not added:  # pragma: no cover
        result += FULL_NOTICE
        added = True
    with path.open("w") as out:
        for line in result:
            out.write(line)


def test_copyright_notices():
    added = check_copyright_notices()
    assert added == 0


if __name__ == "__main__":  # pragma: no cover
    added = check_copyright_notices()
    sys.exit(1 if added else 0)
