from Medical_KG_rev.services.embedding.service import BatchController


def test_choose_respects_override() -> None:
    controller = BatchController()
    controller.reduce("ns", 3)
    assert controller.choose("ns", default=16, pending=10, candidates=[8, 4]) == 3


def test_choose_caps_pending() -> None:
    controller = BatchController()
    size = controller.choose("ns", default=8, pending=3, candidates=[16, 8])
    assert size == 3


def test_history_window_limits_entries() -> None:
    controller = BatchController(window=3)
    for size in [2, 4, 8, 16]:
        controller.record_success("ns", size, 0.1)
    assert len(controller.history["ns"]) == 3
    assert controller.history["ns"][0][0] == 4


def test_reduce_never_below_one() -> None:
    controller = BatchController()
    controller.reduce("ns", 0)
    assert controller.overrides["ns"] == 1


def test_record_success_accepts_zero_duration() -> None:
    controller = BatchController()
    controller.record_success("ns", 8, 0.0)
    assert controller.history["ns"][0][0] == 8
