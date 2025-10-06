from Medical_KG_rev import ping


def test_ping():
    assert ping() == "pong"
