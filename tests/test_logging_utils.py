from src.app.logging_utils import PipelineLog


def test_add_timing_and_total_ms():
    log = PipelineLog()
    log.add_timing("stage_a", 100.0)
    log.add_timing("stage_b", 50.5)
    assert len(log.stage_timings) == 2
    assert log.total_ms() == 150.5


def test_pipeline_log_defaults():
    log = PipelineLog()
    assert log.mode == ""
    assert log.relaxation_steps == []
    assert log.candidate_count == 0
    assert log.total_ms() == 0.0
