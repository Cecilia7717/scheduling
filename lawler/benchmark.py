from pathlib import Path

path = Path("lawler_combined.py")
s = path.read_text()

# Add subprocess-safe timeout support.
s = s.replace(
    "import argparse\n",
    "import argparse\nimport signal\n"
)

# Add retry and resume arguments.
needle = '    parser.add_argument(\n        "--output",'
if needle not in s:
    raise SystemExit("Cannot locate CLI arguments; file unchanged.")

s = s.replace(
    needle,
    '''    parser.add_argument(
        "--timeout", type=float, default=300,
        help="Initial timeout in seconds per solver call"
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Number of additional attempts after timeout"
    )
    parser.add_argument(
        "--timeout-multiplier", type=float, default=2.0,
        help="Multiply timeout after each timeout"
    )
''' + needle,
    1
)

# Add timeout exception and handler.
marker = "def run_lawler("
if marker not in s:
    raise SystemExit("Cannot locate run_lawler; file unchanged.")

s = s.replace(
    marker,
    '''class SolverTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise SolverTimeout("Lawler solver timed out")


def run_with_retries(penalty_jobs, timeout, max_retries,
                     timeout_multiplier):
    current_timeout = timeout

    for attempt in range(max_retries + 1):
        old_handler = signal.signal(
            signal.SIGALRM, _timeout_handler
        )
        start = time.perf_counter()

        try:
            if current_timeout > 0:
                signal.setitimer(
                    signal.ITIMER_REAL, current_timeout
                )

            result = run_lawler(penalty_jobs)
            result["retry_count"] = attempt
            result["effective_timeout"] = current_timeout
            return result

        except SolverTimeout:
            elapsed = time.perf_counter() - start
            print(
                f"  Timeout after {elapsed:.1f}s "
                f"(attempt {attempt + 1}/{max_retries + 1})",
                flush=True
            )

            if attempt == max_retries:
                raise

            current_timeout *= timeout_multiplier

        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)


''' + marker,
    1
)

# Extend result fields.
s = s.replace(
    '    "lawler_runtime_ms",',
    '''    "lawler_runtime_ms",
    "status",
    "retry_count",
    "effective_timeout",
    "error",''',
    1
)

# Add resume support, preserving RNG sequence.
s = s.replace(
    '    rows = []\n\n    master_rng = random.Random(',
    '''    result_path = output_dir / "results.csv"
    rows = []

    if result_path.exists():
        with result_path.open(
            newline="", encoding="utf-8"
        ) as f:
            rows = list(csv.DictReader(f))

    completed_ids = {
        int(row["instance_id"])
        for row in rows
        if row.get("status", "ok") == "ok"
    }

    print(
        f"Resuming: {len(completed_ids)} completed instances",
        flush=True
    )

    master_rng = random.Random(''',
    1
)

# Skip completed instances AFTER consuming structural RNG values,
# so future instance configurations remain reproducible.
needle = '''        kappa_max = master_rng.choice(
            KAPPA_MAX_CHOICES
        )'''
if needle not in s:
    raise SystemExit("Cannot locate random parameter generation.")

s = s.replace(
    needle,
    needle + '''

        if instance_id in completed_ids:
            print(
                f"[{instance_id:03d}/{NUM_INSTANCES:03d}] "
                "Already completed; skipping",
                flush=True
            )
            continue''',
    1
)

# Replace direct solver call with retry wrapper.
needle = '''        result = run_lawler(
            penalty_jobs
        )'''
if needle not in s:
    raise SystemExit("Cannot locate direct solver invocation.")

s = s.replace(
    needle,
    '''        try:
            result = run_with_retries(
                penalty_jobs,
                timeout=args.timeout,
                max_retries=args.max_retries,
                timeout_multiplier=args.timeout_multiplier,
            )
        except SolverTimeout:
            print(
                f"[{instance_id:03d}] All retries timed out; "
                "will retry on the next run",
                flush=True
            )
            continue''',
    1
)

# Add successful-run status.
needle = '''            "lawler_runtime_ms":
                1000.0 * runtime,'''
if needle not in s:
    raise SystemExit("Cannot locate result dictionary ending.")

s = s.replace(
    needle,
    needle + '''

            "status": "ok",
            "retry_count": result["retry_count"],
            "effective_timeout": result["effective_timeout"],
            "error": "",''',
    1
)

# Avoid duplicate records when retrying an old failed instance.
s = s.replace(
    '        rows.append(row)\n\n        # Save after every instance',
    '''        rows = [
            old for old in rows
            if int(old["instance_id"]) != instance_id
        ]
        rows.append(row)

        # Save after every instance''',
    1
)

# Ensure main passes parsed arguments.
s = s.replace(
    'def run_experiment(\n    output_folder: str,\n):',
    'def run_experiment(\n    output_folder: str,\n    args,\n):',
    1
)

s = s.replace(
    '        output_folder=args.output,\n',
    '        output_folder=args.output,\n        args=args,\n',
    1
)

path.write_text(s)
print("Updated", path)