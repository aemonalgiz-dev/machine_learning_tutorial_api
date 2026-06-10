"""The start-up warming names routes that exist, and can be switched off."""

import asyncio

import pytest

from oop_ml_api.main import create_app
from oop_ml_api.warmup import (
    ENVIRONMENT_SWITCH,
    WARM_REQUESTS,
    _status_of_get,
    serves_get,
    warming_enabled,
)


class TestTheList:
    def test_every_request_is_to_a_get_route(self) -> None:
        # A path that was renamed would be warned about at start, which nobody
        # reads, and its page would go back to timing out on a first visit.
        app = create_app()

        missing = [request for request in WARM_REQUESTS if not serves_get(app, request)]

        assert missing == []

    def test_a_path_parameter_is_matched_by_pattern(self) -> None:
        app = create_app()

        assert serves_get(app, "/concepts/searching-a-collection-of-pictures/seeds/1")
        assert not serves_get(app, "/concepts/no-such-page/anything")

    def test_no_request_is_listed_twice(self) -> None:
        assert len(set(WARM_REQUESTS)) == len(WARM_REQUESTS)


class TestTheRequest:
    def test_a_get_goes_through_routing_and_answers(self) -> None:
        assert asyncio.run(_status_of_get(create_app(), "/health")) == 200

    def test_a_query_string_is_validated_like_a_visitor_s(self) -> None:
        # A bad argument is refused by validation rather than reaching the
        # endpoint, which is what makes a listed query string trustworthy.
        status = asyncio.run(
            _status_of_get(
                create_app(),
                "/concepts/convolutional-networks/network?variant=nonsense",
            )
        )

        assert status == 422


class TestTheSwitch:
    def test_on_unless_told_otherwise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENVIRONMENT_SWITCH, raising=False)

        assert warming_enabled()

    def test_a_zero_turns_it_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENVIRONMENT_SWITCH, "0")

        assert not warming_enabled()
