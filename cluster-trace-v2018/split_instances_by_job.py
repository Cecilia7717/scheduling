#!/usr/bin/env python3

import csv
from pathlib import Path
from collections import OrderedDict

DATA_DIR = Path("data")
INSTANCE_FILE = DATA_DIR / "batch_instance.csv"
OUTPUT_DIR = DATA_DIR / "instances_by_job"

# Avoid having thousands of files open simultaneously.
MAX_OPEN_FILES = 100

INSTANCE_COLUMNS = [
    "instance_name",
    "task_name",
    "job_name",
    "task_type",
    "status",
    "start_time",
    "end_time",
    "machine_id",
    "seq_no",
    "total_seq_no",
    "cpu_avg",
    "cpu_max",
    "mem_avg",
    "mem_max",
]


class FileCache:
    def __init__(self, max_open):
        self.max_open = max_open
        self.files = OrderedDict()

    def get_writer(self, job_name):

        if job_name in self.files:
            f, writer = self.files.pop(job_name)
            self.files[job_name] = (f, writer)
            return writer

        # Close least recently used file
        if len(self.files) >= self.max_open:
            _, (old_file, _) = self.files.popitem(last=False)
            old_file.close()

        path = OUTPUT_DIR / f"{job_name}.csv"

        new_file = not path.exists()

        f = open(
            path,
            "a",
            newline="",
            encoding="utf-8",
            buffering=1024 * 1024,
        )

        writer = csv.writer(f)

        if new_file:
            writer.writerow([
                "instance_name",
                "task_name",
                "start_time",
                "end_time",
                "cpu_avg",
            ])

        self.files[job_name] = (f, writer)

        return writer

    def close_all(self):
        for f, _ in self.files.values():
            f.close()

        self.files.clear()


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cache = FileCache(MAX_OPEN_FILES)

    total_rows = 0
    valid_rows = 0

    print(f"Reading: {INSTANCE_FILE}")
    print(f"Writing per-job files to: {OUTPUT_DIR}")
    print()

    try:

        with open(
            INSTANCE_FILE,
            newline="",
            encoding="utf-8",
            buffering=1024 * 1024,
        ) as f:

            reader = csv.reader(f)

            for row in reader:

                total_rows += 1

                if len(row) != len(INSTANCE_COLUMNS):
                    continue

                # Direct indexing is much faster than dict(zip(...))
                instance_name = row[0]
                task_name = row[1]
                job_name = row[2]
                start_time = row[5]
                end_time = row[6]
                cpu_avg = row[10]

                if not job_name:
                    continue

                writer = cache.get_writer(job_name)

                writer.writerow([
                    instance_name,
                    task_name,
                    start_time,
                    end_time,
                    cpu_avg,
                ])

                valid_rows += 1

                if total_rows % 1_000_000 == 0:
                    print(
                        f"Processed {total_rows:,} rows "
                        f"({valid_rows:,} written)"
                    )

    finally:
        cache.close_all()

    print()
    print("Done.")
    print(f"Rows scanned: {total_rows:,}")
    print(f"Rows written: {valid_rows:,}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()