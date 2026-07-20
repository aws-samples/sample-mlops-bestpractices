#!/usr/bin/env python3
"""
Scrub AWS account IDs out of staged files before they land in git.

Used by the pre-commit hook, but also runnable by hand:

    python3 scripts/scrub_account_ids.py            # scrub currently-staged files
    python3 scripts/scrub_account_ids.py --check     # report only, non-zero exit if any found
    python3 scripts/scrub_account_ids.py --all       # scan the whole working tree (tracked files)

What it does
------------
Finds real AWS account IDs and replaces them with the literal placeholder
``<ACCOUNT_ID>``. An account ID is discovered from AWS-specific contexts so we
never clobber unrelated 12-digit numbers (e.g. excalidraw coordinates):

  * ARNs:            arn:aws...:<12 digits>:
  * ECR image URIs:  <12 digits>.dkr.ecr.<region>.amazonaws.com

Every distinct account ID found that way is then replaced *everywhere* in the
same file (including bare occurrences) — safe, because a coordinate can't equal
a string we've already proven is an account ID. As a best effort we also add
the current caller's account ID (via STS) to the scrub set when credentials are
available, so a hardcoded ID with no surrounding ARN still gets caught.

Excluded by design: ``*.ipynb`` (notebook outputs are cleared separately) and
``.env`` (gitignored, auto-populated per user).
"""

import re
import subprocess
import sys

PLACEHOLDER = "<ACCOUNT_ID>"

# Discover account IDs only from unambiguous AWS contexts.
_ARN_RE = re.compile(r"(arn:aws[a-z0-9-]*:[a-z0-9-]*:[a-z0-9-]*:)(\d{12})(:)")
_ECR_RE = re.compile(r"(\d{12})(\.dkr\.ecr\.)")

# Never touch these — notebook outputs are handled by clearing them, .env is
# gitignored and populated per-account.
_SKIP_SUFFIXES = (".ipynb",)
_SKIP_BASENAMES = (".env",)


def _run(args):
    return subprocess.run(
        args, capture_output=True, text=True, check=False
    ).stdout


def staged_files():
    out = _run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACM", "-z"])
    return [p for p in out.split("\0") if p]


def tracked_files():
    out = _run(["git", "ls-files", "-z"])
    return [p for p in out.split("\0") if p]


def _skip(path):
    base = path.rsplit("/", 1)[-1]
    return path.endswith(_SKIP_SUFFIXES) or base in _SKIP_BASENAMES


def _read_text(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
        return None  # binary / gone / dir — skip


def _caller_account_id():
    """Best-effort: the current AWS caller's account ID, or None."""
    try:
        import boto3  # noqa: local import — optional dependency at commit time
        return boto3.client("sts").get_caller_identity()["Account"]
    except Exception:
        return None


def scrub_text(text, extra_ids=()):
    """Return (scrubbed_text, sorted_ids_found)."""
    found = set()
    found.update(m.group(2) for m in _ARN_RE.finditer(text))
    found.update(m.group(1) for m in _ECR_RE.finditer(text))

    # Replace bare occurrences of every ID we're confident about.
    for acct in sorted(found | {i for i in extra_ids if i and i in text}):
        if acct in text:
            found.add(acct)
            text = text.replace(acct, PLACEHOLDER)

    # Belt-and-suspenders: rewrite the account slot inside any ARN/ECR pattern
    # that somehow still carries digits (e.g. an ID not caught as bare above).
    text = _ARN_RE.sub(lambda m: m.group(1) + PLACEHOLDER + m.group(3), text)
    text = _ECR_RE.sub(lambda m: PLACEHOLDER + m.group(2), text)
    return text, sorted(found)


def main(argv):
    check_only = "--check" in argv
    scan_all = "--all" in argv

    files = tracked_files() if scan_all else staged_files()
    caller = _caller_account_id()
    extra = {caller} if caller else set()

    changed, findings = [], {}
    for path in files:
        if _skip(path):
            continue
        text = _read_text(path)
        if text is None:
            continue
        scrubbed, ids = scrub_text(text, extra_ids=extra)
        if scrubbed != text:
            findings[path] = ids
            if not check_only:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(scrubbed)
                subprocess.run(["git", "add", "--", path], check=False)
            changed.append(path)

    if changed:
        verb = "would scrub" if check_only else "scrubbed"
        print(f"account-id scrub: {verb} {len(changed)} file(s):", file=sys.stderr)
        for path in changed:
            ids = ", ".join(findings.get(path, [])) or "(context-only)"
            print(f"  - {path}  [{ids}]", file=sys.stderr)
        if check_only:
            print(
                "Run: python3 scripts/scrub_account_ids.py  (or re-stage) to fix.",
                file=sys.stderr,
            )
            return 1
        print("Re-staged scrubbed files; commit will use the scrubbed content.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
