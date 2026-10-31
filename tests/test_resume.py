import json

from arachne_small.trainer import _truncate_training_log


def test_resume_truncates_records_after_checkpoint_and_drops_partial_tail(tmp_path):
    path = tmp_path / "training.jsonl"
    path.write_text(
        json.dumps({"step": 1, "loss": 2.0}) + "\n"
        + json.dumps({"step": 2, "loss": 1.8}) + "\n"
        + '{"step":3,"loss":',
        encoding="utf-8",
    )

    _truncate_training_log(path, checkpoint_step=1)

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records == [{"step": 1, "loss": 2.0}]
