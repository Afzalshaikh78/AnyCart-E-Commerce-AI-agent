from Backend.api.metrics import elapsed_ms


def test_elapsed_ms_is_positive() -> None:
    assert elapsed_ms(0) > 0
