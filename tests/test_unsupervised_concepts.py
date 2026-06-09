"""The k-means and PCA endpoints, pinned to the pages' hand arithmetic.

The k-means page works eight people in two tight groups whose resting centres
are the plain group means, (121, 26) and (181, 80), with an inertia of exactly
64 summed by hand. The PCA page works four people whose deviation sums build
the matrix [[250, 150], [150, 250]], whose kept directions are the linear
algebra primer's (1, 1) and (1, -1) with shares of 0.8 and 0.2. All pinned.
"""

from itertools import pairwise
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from oop_ml_api.main import create_app

client = TestClient(create_app())

KMEANS = "/concepts/k-means/cluster"
PCA = "/concepts/pca/analyze"

TWO_GROUPS = [
    {"x": 118, "y": 24},
    {"x": 120, "y": 25},
    {"x": 122, "y": 28},
    {"x": 124, "y": 27},
    {"x": 178, "y": 78},
    {"x": 180, "y": 80},
    {"x": 182, "y": 83},
    {"x": 184, "y": 79},
]

FOUR_PEOPLE = [
    {"x": 180, "y": 78},
    {"x": 160, "y": 58},
    {"x": 175, "y": 63},
    {"x": 165, "y": 73},
]


class TestKMeans:
    def test_the_resting_centres_are_the_group_means(self) -> None:
        body = client.post(KMEANS, json={"points": TWO_GROUPS, "k": 2}).json()

        centres = sorted([(each["x"], each["y"]) for each in body["centres"]])
        assert centres == [(121.0, 26.0), (181.0, 80.0)]

    def test_the_inertia_is_the_hand_worked_sixty_four(self) -> None:
        body = client.post(KMEANS, json={"points": TWO_GROUPS, "k": 2}).json()

        assert body["inertia"] == 64.0

    def test_the_grouping_separates_the_two_clumps(self) -> None:
        body = client.post(KMEANS, json={"points": TWO_GROUPS, "k": 2}).json()

        labels = body["labels"]
        assert len(set(labels[:4])) == 1
        assert len(set(labels[4:])) == 1
        assert labels[0] != labels[4]

    def test_one_group_holds_everyone(self) -> None:
        body = client.post(KMEANS, json={"points": TWO_GROUPS, "k": 1}).json()

        assert set(body["labels"]) == {0}
        assert len(body["centres"]) == 1

    def test_inertia_never_rises_with_more_groups(self) -> None:
        inertias = [
            client.post(KMEANS, json={"points": TWO_GROUPS, "k": k}).json()["inertia"]
            for k in (1, 2, 4)
        ]

        assert inertias[0] >= inertias[1] >= inertias[2]

    def test_more_groups_than_people_is_refused_in_words(self) -> None:
        response = client.post(KMEANS, json={"points": TWO_GROUPS[:3], "k": 5})

        assert response.status_code == 422
        assert "conjured" in response.json()["detail"]


class TestPca:
    def test_the_directions_are_the_primer_pair(self) -> None:
        body = client.post(PCA, json={"points": FOUR_PEOPLE}).json()

        first, second = body["components"]
        # The long direction lies along (1, 1) and the short along (1, -1),
        # reported at unit length.
        assert abs(abs(first["dx"]) - abs(first["dy"])) < 1e-9
        assert abs(abs(second["dx"]) - abs(second["dy"])) < 1e-9
        assert abs(first["dx"] * first["dy"]) > 0.49  # same signs
        assert second["dx"] * second["dy"] < 0  # opposite signs

    def test_the_shares_are_four_fifths_and_one_fifth(self) -> None:
        body = client.post(PCA, json={"points": FOUR_PEOPLE}).json()

        first, second = body["components"]
        assert abs(first["share"] - 0.8) < 1e-9
        assert abs(second["share"] - 0.2) < 1e-9

    def test_the_mean_is_the_cloud_centre(self) -> None:
        body = client.post(PCA, json={"points": FOUR_PEOPLE}).json()

        assert body["mean"] == {"x": 170.0, "y": 68.0}

    def test_reconstructions_sit_on_the_long_diagonal(self) -> None:
        # Flattened to one component, every rebuilt person satisfies
        # weight - 68 = height - 70, the line through the mean along (1, 1).
        body = client.post(PCA, json={"points": FOUR_PEOPLE}).json()

        for point in body["reconstructions"]:
            assert abs((point["y"] - 68.0) - (point["x"] - 170.0)) < 1e-9

    def test_a_lone_point_is_refused(self) -> None:
        response = client.post(PCA, json={"points": [{"x": 170, "y": 68}]})

        assert response.status_code == 422


class TestInertiaCurve:
    """The staircase the k-means chart draws, one pass budget at a time."""

    UNLABELLED_CROWD: ClassVar[list[dict]] = [
        {"x": 147.0, "y": 41.0},
        {"x": 156.0, "y": 53.0},
        {"x": 145.0, "y": 57.0},
        {"x": 159.0, "y": 57.0},
        {"x": 162.0, "y": 61.0},
        {"x": 120.0, "y": 25.0},
        {"x": 122.0, "y": 28.0},
        {"x": 118.0, "y": 24.0},
        {"x": 180.0, "y": 80.0},
        {"x": 183.0, "y": 83.0},
        {"x": 178.0, "y": 78.0},
    ]

    def test_the_total_only_falls_or_holds_still(self) -> None:
        response = client.post(
            "/concepts/k-means/inertia-curve",
            json={"points": self.UNLABELLED_CROWD, "n_clusters": 2},
        )

        inertias = response.json()["inertias"]
        assert len(inertias) == 12
        assert all(later <= earlier + 1e-9 for earlier, later in pairwise(inertias))

    def test_the_walk_comes_to_rest(self) -> None:
        response = client.post(
            "/concepts/k-means/inertia-curve",
            json={"points": self.UNLABELLED_CROWD, "n_clusters": 2},
        )

        inertias = response.json()["inertias"]
        assert inertias[-1] == inertias[-2]


class TestVarianceSweep:
    """The half-turn of variances the PCA chart draws."""

    def sweep(self) -> dict:
        response = client.post(
            "/concepts/pca/variance-sweep",
            json={
                "points": [
                    {"x": float(x), "y": float(0.8 * x - 68.0 + noise)}
                    for x, noise in zip(
                        range(150, 190, 2),
                        [
                            3,
                            -2,
                            1,
                            -4,
                            2,
                            0,
                            -1,
                            3,
                            -3,
                            1,
                            2,
                            -2,
                            0,
                            4,
                            -1,
                            1,
                            -3,
                            2,
                            0,
                            -2,
                        ],
                        strict=True,
                    )
                ]
            },
        )
        assert response.status_code == 200
        return response.json()

    def test_the_peak_is_the_first_component(self) -> None:
        body = self.sweep()
        best = max(body["components"], key=lambda entry: entry["variance"])
        assert max(body["variances"]) <= best["variance"] + 1e-9

    def test_the_curve_passes_through_both_components(self) -> None:
        body = self.sweep()
        for component in body["components"]:
            nearest = min(
                zip(body["angles_degrees"], body["variances"], strict=True),
                key=lambda pair: abs(pair[0] - component["angle_degrees"]),
            )
            assert abs(nearest[1] - component["variance"]) < 0.5

    def test_the_directions_are_a_quarter_turn_apart(self) -> None:
        body = self.sweep()
        first, second = body["components"]
        separation = abs(first["angle_degrees"] - second["angle_degrees"])
        assert round(min(separation, 180.0 - separation), 6) == 90.0


# --- The k-means page's further endpoints, pinned to what the page quotes ----

WALK = "/concepts/k-means/walk"
RESTARTS = "/concepts/k-means/restarts"
ELBOW = "/concepts/k-means/elbow"
OTHERS = "/concepts/k-means/other-definitions"
UNITS = "/concepts/k-means/units"
CENTRE = "/concepts/k-means/centre-choice"

UNLABELLED_CROWD = [
    {"x": 147, "y": 41},
    {"x": 156, "y": 53},
    {"x": 145, "y": 57},
    {"x": 159, "y": 57},
    {"x": 162, "y": 61},
    {"x": 120, "y": 25},
    {"x": 122, "y": 28},
    {"x": 118, "y": 24},
    {"x": 180, "y": 80},
    {"x": 183, "y": 83},
    {"x": 178, "y": 78},
]

BAND_HEIGHTS = list(range(120, 191, 7))
TWO_BANDS = [
    {"x": height, "y": round(25 + 0.6 * (height - 120), 1)} for height in BAND_HEIGHTS
] + [{"x": height, "y": round(39 + 0.6 * (height - 120), 1)} for height in BAND_HEIGHTS]

TWO_CRESCENTS = [
    {"x": x, "y": y}
    for x, y in [
        (173, 58),
        (171, 67),
        (167, 74),
        (160, 80),
        (152, 83),
        (144, 83),
        (136, 80),
        (129, 74),
        (125, 67),
        (123, 58),
        (148, 46),
        (150, 37),
        (154, 29),
        (160, 24),
        (169, 21),
        (177, 21),
        (186, 24),
        (192, 29),
        (196, 37),
        (198, 45),
    ]
]

STRAY = {"x": 196, "y": 14}
WITH_A_STRAY = [*TWO_GROUPS, STRAY]


def _partition(labels: list[int]) -> set[frozenset[int]]:
    groups: dict[int, set[int]] = {}
    for row, label in enumerate(labels):
        groups.setdefault(label, set()).add(row)
    return {frozenset(rows) for rows in groups.values()}


class TestClusterSizes:
    def test_the_two_groups_hold_four_each(self) -> None:
        body = client.post(KMEANS, json={"points": TWO_GROUPS, "k": 2}).json()

        assert sorted(body["sizes"]) == [4, 4]

    def test_a_stray_person_drags_a_centre_off_its_group(self) -> None:
        body = client.post(KMEANS, json={"points": WITH_A_STRAY, "k": 2}).json()

        centres = sorted((each["x"], each["y"]) for each in body["centres"])
        assert centres == [(121.0, 26.0), (184.0, 66.8)]
        assert body["inertia"] == pytest.approx(3728.8)
        assert body["labels"][-1] == body["labels"][4]

    def test_two_bands_are_cut_across_rather_than_along(self) -> None:
        body = client.post(KMEANS, json={"points": TWO_BANDS, "k": 2}).json()

        assert body["inertia"] == pytest.approx(4504.436363636363)
        assert sorted(body["sizes"]) == [11, 11]
        # The lower band's first six people share a group with the upper
        # band's first five, which is a cut by height, not by band.
        assert len(set(body["labels"][:6] + body["labels"][11:16])) == 1
        assert body["labels"][0] != body["labels"][6]


class TestWalk:
    """One seeded start on the crowd, watched pass by pass."""

    def walk(self, points: list[dict], k: int, seed: int | None = None) -> dict:
        body = {"points": points, "k": k}
        if seed is not None:
            body["seed"] = seed
        response = client.post(WALK, json=body)
        assert response.status_code == 200
        return response.json()

    def test_the_seeded_start_is_two_of_the_people(self) -> None:
        body = self.walk(UNLABELLED_CROWD, 2)

        starts = [(each["x"], each["y"]) for each in body["start"]["centres"]]
        people = [(each["x"], each["y"]) for each in UNLABELLED_CROWD]
        assert all(start in people for start in starts)
        assert body["start"]["inertia"] == pytest.approx(8197.0)

    def test_every_half_step_lowers_the_total_or_holds_it(self) -> None:
        body = self.walk(UNLABELLED_CROWD, 2)

        readings = [body["start"]["inertia"]]
        for each in body["passes"]:
            readings.append(each["inertia_after_update"])
            readings.append(each["inertia_after_assign"])
        assert all(later <= earlier + 1e-9 for earlier, later in pairwise(readings))

    def test_the_update_step_moves_each_centre_to_the_mean_of_who_chose_it(
        self,
    ) -> None:
        body = self.walk(UNLABELLED_CROWD, 2)

        first = body["passes"][0]
        for group, centre in enumerate(first["centres"]):
            members = [
                person
                for person, label in zip(UNLABELLED_CROWD, first["labels_used"])
                if label == group
            ]
            assert centre["x"] == pytest.approx(
                sum(person["x"] for person in members) / len(members)
            )
            assert centre["y"] == pytest.approx(
                sum(person["y"] for person in members) / len(members)
            )

    def test_seed_three_rests_in_the_worse_valley_after_five_passes(self) -> None:
        body = self.walk(UNLABELLED_CROWD, 2)

        assert body["settled_pass"] == 5
        assert body["passes"][-1]["largest_shift"] == 0.0
        assert body["resting_inertia"] == pytest.approx(3185.8666666666677)
        assert body["best_of_ten_inertia"] == pytest.approx(2952.6071428571413)
        assert body["rests_in_the_best_valley"] is False
        after_assign = [each["inertia_after_assign"] for each in body["passes"]]
        assert after_assign[:4] == pytest.approx(
            [
                4086.7690972222217,
                3724.6033163265297,
                3425.6222222222223,
                3185.8666666666677,
            ]
        )

    def test_seed_one_reaches_the_best_valley_in_one_pass(self) -> None:
        body = self.walk(UNLABELLED_CROWD, 2, seed=1)

        assert body["start"]["inertia"] == pytest.approx(5047.0)
        assert body["passes"][0]["inertia_after_assign"] == pytest.approx(
            2952.6071428571413
        )
        assert body["settled_pass"] == 2
        assert body["rests_in_the_best_valley"] is True

    def test_the_two_groups_settle_in_one_update(self) -> None:
        body = self.walk(TWO_GROUPS, 2)

        assert body["start"]["inertia"] == pytest.approx(156.0)
        assert body["passes"][0]["inertia_after_update"] == pytest.approx(64.0)
        assert body["passes"][0]["largest_shift"] == pytest.approx(13.0)
        assert body["settled_pass"] == 2

    def test_more_groups_than_people_is_refused_in_words(self) -> None:
        response = client.post(WALK, json={"points": TWO_GROUPS[:3], "k": 5})

        assert response.status_code == 422
        assert "conjured" in response.json()["detail"]


class TestRestarts:
    """The same people from ten seeds, and which valleys the seeds find."""

    def test_the_crowd_at_two_groups_has_three_valleys(self) -> None:
        body = client.post(RESTARTS, json={"points": UNLABELLED_CROWD, "k": 2}).json()

        inertias = [round(each["inertia"], 3) for each in body["fits"]]
        assert body["distinct_inertias"] == 3
        assert inertias.count(2952.607) == 5
        assert inertias.count(3185.867) == 3
        assert inertias.count(3141.667) == 2
        assert body["best_seed"] == 1
        assert body["best_inertia"] == pytest.approx(2952.6071428571413)

    def test_the_library_keeps_the_best_of_the_same_ten(self) -> None:
        body = client.post(RESTARTS, json={"points": UNLABELLED_CROWD, "k": 2}).json()

        assert body["with_restarts_inertia"] == pytest.approx(body["best_inertia"])
        assert sum(each["same_grouping_as_best"] for each in body["fits"]) == 5

    def test_the_crowd_at_three_groups_has_one_valley(self) -> None:
        body = client.post(RESTARTS, json={"points": UNLABELLED_CROWD, "k": 3}).json()

        assert body["distinct_inertias"] == 1
        assert all(each["same_grouping_as_best"] for each in body["fits"])
        assert all(sorted(each["sizes"]) == [3, 3, 5] for each in body["fits"])

    def test_restarts_do_not_rescue_the_two_bands(self) -> None:
        body = client.post(RESTARTS, json={"points": TWO_BANDS, "k": 2}).json()

        assert body["distinct_inertias"] == 1
        assert all(each["same_grouping_as_best"] for each in body["fits"])


class TestElbow:
    def test_the_two_groups_fall_from_the_total_scatter_to_nothing(self) -> None:
        body = client.post(ELBOW, json={"points": TWO_GROUPS, "max_k": 8}).json()

        assert body["total_scatter"] == pytest.approx(13096.0)
        assert [each["inertia"] for each in body["points"]] == pytest.approx(
            [13096.0, 64.0, 39.0, 19.0, 9.0, 5.0, 2.5, 0.0]
        )
        assert body["points"][0]["drop_from_previous"] is None
        assert body["points"][1]["drop_from_previous"] == pytest.approx(13032.0)

    def test_the_crowd_keeps_falling_past_its_three_clumps(self) -> None:
        body = client.post(ELBOW, json={"points": UNLABELLED_CROWD, "max_k": 6}).json()

        inertias = [each["inertia"] for each in body["points"]]
        assert inertias == pytest.approx(
            [10482.181818181818, 2952.6071428571413, 501.6, 222.0, 92.0, 54.5]
        )
        assert all(later <= earlier for earlier, later in pairwise(inertias))
        assert body["points"][2]["share_of_total"] == pytest.approx(0.0479, abs=1e-4)

    def test_the_sweep_stops_at_the_number_of_people(self) -> None:
        body = client.post(ELBOW, json={"points": TWO_GROUPS[:3], "max_k": 8}).json()

        assert [each["k"] for each in body["points"]] == [1, 2, 3]
        assert body["points"][-1]["inertia"] == pytest.approx(0.0)


class TestOtherDefinitions:
    def grouped(self, points: list[dict], k: int, radius: float) -> dict:
        response = client.post(
            OTHERS,
            json={
                "points": points,
                "k": k,
                "radius": radius,
                "min_neighbourhood_size": 2,
            },
        )
        assert response.status_code == 200
        return response.json()

    def test_density_and_single_linkage_follow_the_bands_and_the_mixture_does_not(
        self,
    ) -> None:
        body = self.grouped(TWO_BANDS, 2, 10.0)

        bands = _partition([0] * 11 + [1] * 11)
        assert _partition(body["density"]["labels"]) == bands
        assert body["density"]["n_noise"] == 0
        assert _partition(body["single_linkage"]["labels"]) == bands
        assert _partition(body["k_means"]["labels"]) != bands
        assert _partition(body["mixture"]["labels"]) == _partition(
            body["k_means"]["labels"]
        )

    def test_three_of_the_four_follow_the_crescents(self) -> None:
        body = self.grouped(TWO_CRESCENTS, 2, 10.0)

        arcs = _partition([0] * 10 + [1] * 10)
        assert _partition(body["density"]["labels"]) == arcs
        assert _partition(body["single_linkage"]["labels"]) == arcs
        assert _partition(body["mixture"]["labels"]) == arcs
        assert _partition(body["k_means"]["labels"]) != arcs
        assert body["k_means"]["inertia"] == pytest.approx(7949.1)

    def test_the_density_grouping_leaves_the_stray_in_no_group(self) -> None:
        body = self.grouped(WITH_A_STRAY, 2, 6.0)

        assert body["density"]["labels"][-1] == -1
        assert body["density"]["n_noise"] == 1
        assert body["density"]["n_clusters"] == 2
        assert body["k_means"]["labels"][-1] == body["k_means"]["labels"][4]


class TestUnits:
    def test_millimetres_change_the_two_group_answer_and_metres_do_not(self) -> None:
        body = client.post(UNITS, json={"points": UNLABELLED_CROWD, "k": 2}).json()

        by_unit = {each["unit"]: each for each in body["variants"]}
        assert by_unit["centimetres"]["same_grouping_as_centimetres"] is True
        assert by_unit["metres"]["same_grouping_as_centimetres"] is True
        assert by_unit["millimetres"]["same_grouping_as_centimetres"] is False
        assert sorted(by_unit["millimetres"]["sizes"]) == [5, 6]
        assert body["standardised"]["same_grouping_as_centimetres"] is True

    def test_metres_change_the_three_group_answer_and_millimetres_do_not(self) -> None:
        body = client.post(UNITS, json={"points": UNLABELLED_CROWD, "k": 3}).json()

        by_unit = {each["unit"]: each for each in body["variants"]}
        assert by_unit["metres"]["same_grouping_as_centimetres"] is False
        assert sorted(by_unit["metres"]["sizes"]) == [3, 4, 4]
        assert by_unit["millimetres"]["same_grouping_as_centimetres"] is True
        assert body["standardised"]["same_grouping_as_centimetres"] is True

    def test_the_standardised_centres_come_back_in_the_original_units(self) -> None:
        body = client.post(UNITS, json={"points": UNLABELLED_CROWD, "k": 3}).json()

        centres = sorted(
            (round(each["x"], 3), round(each["y"], 3))
            for each in body["standardised"]["centres"]
        )
        assert centres == [(120.0, 25.667), (153.8, 53.8), (180.333, 80.333)]


class TestCentreChoice:
    def test_the_mean_wins_squared_distance_and_the_median_wins_absolute(self) -> None:
        body = client.post(CENTRE, json={"points": [*TWO_GROUPS[4:], STRAY]}).json()

        assert body["mean"] == {"x": 184.0, "y": 66.8}
        assert body["median"] == {"x": 182.0, "y": 79.0}
        assert body["squared_at_mean"] == pytest.approx(3698.8)
        assert body["squared_at_median"] == pytest.approx(4463.0)
        assert body["absolute_at_mean"] == pytest.approx(129.6)
        assert body["absolute_at_median"] == pytest.approx(93.0)

    def test_on_the_tall_four_alone_the_mean_still_wins_squared_distance(self) -> None:
        body = client.post(CENTRE, json={"points": TWO_GROUPS[4:]}).json()

        assert body["squared_at_mean"] == pytest.approx(34.0)
        assert body["squared_at_median"] == pytest.approx(35.0)
        assert body["absolute_at_mean"] == pytest.approx(body["absolute_at_median"])


class TestRefusals:
    @pytest.mark.parametrize("path", [RESTARTS, UNITS])
    def test_more_groups_than_people_is_refused_in_words(self, path: str) -> None:
        response = client.post(path, json={"points": TWO_GROUPS[:3], "k": 5})

        assert response.status_code == 422
        assert "conjured" in response.json()["detail"]

    def test_an_unknown_field_is_refused(self) -> None:
        response = client.post(WALK, json={"points": TWO_GROUPS, "k": 2, "seeds": 3})

        assert response.status_code == 422

    def test_identical_people_leave_a_group_empty_rather_than_failing(self) -> None:
        body = client.post(
            KMEANS, json={"points": [{"x": 150, "y": 50}] * 6, "k": 2}
        ).json()

        assert sorted(body["sizes"]) == [0, 6]
        assert body["inertia"] == 0.0

    def test_a_constant_column_is_accepted(self) -> None:
        flat = [{"x": 150, "y": point["y"]} for point in TWO_GROUPS]
        body = client.post(KMEANS, json={"points": flat, "k": 2}).json()

        assert body["inertia"] == pytest.approx(24.0)
        assert sorted(body["sizes"]) == [4, 4]


LOOSE_TWENTY = [
    {"x": x, "y": y}
    for x, y in [
        (136, 48),
        (139, 61),
        (141, 52),
        (143, 44),
        (145, 66),
        (147, 57),
        (149, 49),
        (150, 62),
        (152, 54),
        (154, 45),
        (155, 68),
        (157, 59),
        (158, 50),
        (160, 64),
        (162, 55),
        (163, 47),
        (165, 60),
        (164, 69),
        (146, 43),
        (138, 55),
    ]
]
TIGHT_FIVE = [
    {"x": x, "y": y} for x, y in [(176, 56), (178, 60), (180, 58), (177, 59), (179, 57)]
]
UNEQUAL_SIZES = [*LOOSE_TWENTY, *TIGHT_FIVE]


class TestUnequalSizes:
    def test_two_groups_cut_the_loose_twenty_and_absorb_the_tight_five(self) -> None:
        body = client.post(KMEANS, json={"points": UNEQUAL_SIZES, "k": 2}).json()

        assert body["inertia"] == pytest.approx(2464.633, abs=1e-3)
        assert sorted(body["sizes"]) == [10, 15]
        assert len(set(body["labels"][20:])) == 1
        assert body["labels"][:20].count(body["labels"][20]) == 5

    def test_three_groups_give_the_five_their_own_centre(self) -> None:
        body = client.post(KMEANS, json={"points": UNEQUAL_SIZES, "k": 3}).json()

        assert body["inertia"] == pytest.approx(1537.495, abs=1e-3)
        assert sorted(body["sizes"]) == [5, 9, 11]
        assert len(set(body["labels"][20:])) == 1
        assert body["labels"][20] not in body["labels"][:20]

    def test_only_three_of_ten_starts_find_that_valley(self) -> None:
        body = client.post(RESTARTS, json={"points": UNEQUAL_SIZES, "k": 3}).json()

        assert body["distinct_inertias"] == 7
        assert sum(each["same_grouping_as_best"] for each in body["fits"]) == 3

    def test_the_other_three_definitions_separate_the_five_at_two_groups(self) -> None:
        body = client.post(
            OTHERS,
            json={
                "points": UNEQUAL_SIZES,
                "k": 2,
                "radius": 10.0,
                "min_neighbourhood_size": 2,
            },
        ).json()

        split = _partition([0] * 20 + [1] * 5)
        assert _partition(body["density"]["labels"]) == split
        assert _partition(body["single_linkage"]["labels"]) == split
        assert _partition(body["mixture"]["labels"]) == split
        assert _partition(body["k_means"]["labels"]) != split
