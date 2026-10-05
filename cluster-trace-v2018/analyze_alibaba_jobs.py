#!/usr/bin/env python3

import csv
from pathlib import Path


DATA_DIR = Path("data")
TASK_FILE = DATA_DIR / "batch_task.csv"
OUTPUT_FILE = Path("all_job_stats.csv")

TASK_COLUMNS = [
    "task_name",
    "instance_num",
    "job_name",
    "task_type",
    "status",
    "start_time",
    "end_time",
    "plan_cpu",
    "plan_mem",
]


def safe_int(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def write_job(writer, job_name, tasks):
    """
    Compute lightweight statistics for one Alibaba job using
    batch_task.csv only.
    """

    if not tasks:
        return False

    num_tasks = len(tasks)

    total_instances = 0
    start_times = []
    end_times = []

    valid_task_rows = 0
    invalid_task_rows = 0

    for row in tasks:

        instance_num = safe_int(row[1])
        start_time = safe_int(row[5])
        end_time = safe_int(row[6])

        if (
            instance_num is None
            or start_time is None
            or end_time is None
        ):
            invalid_task_rows += 1
            continue

        if instance_num < 0 or end_time < start_time:
            invalid_task_rows += 1
            continue

        total_instances += instance_num
        start_times.append(start_time)
        end_times.append(end_time)

        valid_task_rows += 1

    if not start_times or not end_times:
        job_start = ""
        job_end = ""
        horizon = ""
    else:
        job_start = min(start_times)
        job_end = max(end_times)
        horizon = job_end - job_start

    writer.writerow([
        job_name,
        num_tasks,
        total_instances,
        job_start,
        job_end,
        horizon,
        valid_task_rows,
        invalid_task_rows,
    ])

    return True


def main():

    print(f"Reading: {TASK_FILE}")
    print(f"Writing: {OUTPUT_FILE}")
    print()

    total_rows = 0
    total_jobs = 0

    current_job = None
    current_tasks = []

    with open(
        TASK_FILE,
        newline="",
        encoding="utf-8",
        buffering=1024 * 1024,
    ) as input_f, open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
        buffering=1024 * 1024,
    ) as output_f:

        reader = csv.reader(input_f)
        writer = csv.writer(output_f)

        writer.writerow([
            "job_name",
            "num_tasks",
            "total_instances",
            "start_time",
            "end_time",
            "horizon",
            "valid_task_rows",
            "invalid_task_rows",
        ])

        for row in reader:

            total_rows += 1

            if len(row) != len(TASK_COLUMNS):
                continue

            job_name = row[2]

            if current_job is None:
                current_job = job_name

            # New job encountered
            if job_name != current_job:

                if write_job(
                    writer,
                    current_job,
                    current_tasks
                ):
                    total_jobs += 1

                if total_jobs % 100_000 == 0:
                    print(
                        f"Processed "
                        f"{total_jobs:,} jobs, "
                        f"{total_rows:,} task rows"
                    )

                current_job = job_name
                current_tasks = []

            current_tasks.append(row)

        # Final job
        if current_job is not None:

            if write_job(
                writer,
                current_job,
                current_tasks
            ):
                total_jobs += 1

    print()
    print("Done.")
    print(f"Task rows read: {total_rows:,}")
    print(f"Jobs written:   {total_jobs:,}")
    print(f"Output:         {OUTPUT_FILE}")


if __name__ == "__main__":
    main()