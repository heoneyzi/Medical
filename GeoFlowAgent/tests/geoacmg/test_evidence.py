"""The point scale is the benchmark's notion of progress, so it is checked hard."""

from __future__ import annotations

import pytest

from geoflowagent.geoacmg.evidence import (
    CRITERIA,
    Classification,
    Strength,
    UnknownEvidenceCode,
    band,
    band_of_assertion,
    parse_code,
    parse_codes,
    points_to_band,
    reconstruct,
    score,
)


class TestCodeGrammar:
    @pytest.mark.parametrize(
        "text,base,strength,points",
        [
            ("PVS1", "PVS1", Strength.VERY_STRONG, 8),
            ("PM2", "PM2", Strength.MODERATE, 2),
            ("PM2_Supporting", "PM2", Strength.SUPPORTING, 1),
            ("PP1_Strong", "PP1", Strength.STRONG, 4),
            ("PS4_Very Strong", "PS4", Strength.VERY_STRONG, 8),
            ("BA1", "BA1", Strength.STAND_ALONE, -8),
            ("BS2_Stand Alone", "BS2", Strength.STAND_ALONE, -8),
            ("BP4", "BP4", Strength.SUPPORTING, -1),
        ],
    )
    def test_published_spellings(self, text, base, strength, points):
        code = parse_code(text)
        assert (code.base, code.strength, code.points) == (base, strength, points)

    @pytest.mark.parametrize("text", ["PM3_Very", "BS1_Stand"])
    def test_truncated_spellings_are_normalised_not_dropped(self, text):
        """The published export writes both forms. Dropping them loses 320 records."""

        assert parse_code(text).strength in {Strength.VERY_STRONG, Strength.STAND_ALONE}

    @pytest.mark.parametrize("text", ["", "PX9", "PM2_Enormous", "hello"])
    def test_unknown_input_raises_rather_than_scoring_zero(self, text):
        with pytest.raises(UnknownEvidenceCode):
            parse_code(text)

    def test_every_criterion_round_trips_through_its_label(self):
        for base in CRITERIA:
            code = parse_code(base)
            assert parse_code(code.label) == code


class TestPointScale:
    def test_canonical_pathogenic_combination(self):
        """PVS1 + PM2_Supporting + PP3 = 8+1+1 = 10, the Pathogenic boundary."""

        points, classification = reconstruct(parse_codes("PVS1,PM2_Supporting,PP3"))
        assert points == 10
        assert classification is Classification.PATHOGENIC

    def test_stand_alone_benign_needs_nothing_else(self):
        assert reconstruct(parse_codes("BA1"))[1] is Classification.BENIGN

    def test_no_evidence_is_uncertain(self):
        assert band(score([])) is Classification.UNCERTAIN

    @pytest.mark.parametrize(
        "points,expected",
        [(10, Classification.PATHOGENIC), (9, Classification.LIKELY_PATHOGENIC),
         (6, Classification.LIKELY_PATHOGENIC), (5, Classification.UNCERTAIN),
         (0, Classification.UNCERTAIN), (-1, Classification.LIKELY_BENIGN),
         (-6, Classification.LIKELY_BENIGN), (-7, Classification.BENIGN)],
    )
    def test_band_boundaries_are_exact(self, points, expected):
        assert band(points) is expected

    def test_bands_tile_the_line_without_gaps_or_overlap(self):
        for points in range(-40, 41):
            assert band(points) is not None

    def test_distance_to_band_is_zero_inside_and_positive_outside(self):
        assert points_to_band(12, Classification.PATHOGENIC) == 0
        assert points_to_band(6, Classification.PATHOGENIC) == 4
        assert points_to_band(-3, Classification.PATHOGENIC) == 13

    def test_assertion_strings_map_onto_bands(self):
        assert band_of_assertion("Likely Pathogenic") is Classification.LIKELY_PATHOGENIC
        assert band_of_assertion("uncertain significance") is Classification.UNCERTAIN
        with pytest.raises(UnknownEvidenceCode):
            band_of_assertion("Probably Fine")
