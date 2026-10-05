from pathlib import Path

import processe_data


def test_process_task_writes_task_row(tmp_path):
    task_file_cache = processe_data.FileHandleCache(max_open=16)
    row = {
        "task_name": "M1",
        "instance_num": "7",
        "job_name": "job_42",
        "task_type": "1",
        "status": "SUCCESS",
        "start_time": "10",
        "end_time": "20",
        "plan_cpu": "1.0",
        "plan_mem": "2.0",
    }

    job_dir = tmp_path / row["job_name"]
    processe_data.process_task(row, task_file_cache=task_file_cache, job_dir=job_dir)

    tasks_csv = job_dir / "tasks.csv"
    assert tasks_csv.exists()

    with tasks_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(__import__("csv").reader(f))

    assert rows[0][0] == "M1"
    assert rows[0][1] == "1"
    assert rows[0][2] == ""
    assert rows[0][3] == "7"
    assert rows[0][8] == "1"
    assert rows[0][9] == "SUCCESS"

    task_file_cache.close_all()
