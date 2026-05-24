from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nys_aq.daily import (  # noqa: E402
    DailyConfig,
    _upsert_daily_csv,
    point_in_multipolygon,
    prune_old_artifacts,
    write_outputs,
    stable_sample_locations,
)


class StableSampleLocationsTests(unittest.TestCase):
    def test_is_deterministic_and_respects_limit(self) -> None:
        locations = [{"id": i} for i in range(1, 8)]

        first = stable_sample_locations(locations, 3, salt="sample-salt")
        second = stable_sample_locations(locations, 3, salt="sample-salt")

        self.assertEqual([loc["id"] for loc in first], [loc["id"] for loc in second])
        self.assertEqual(len(first), 3)
        self.assertTrue(all(loc in locations for loc in first))


class GeometryTests(unittest.TestCase):
    def test_point_in_multipolygon_uses_outer_ring(self) -> None:
        multipolygon = [
            [
                [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]],
            ]
        ]

        self.assertTrue(point_in_multipolygon(2.0, 2.0, multipolygon))
        self.assertFalse(point_in_multipolygon(5.0, 2.0, multipolygon))


class CsvUpsertTests(unittest.TestCase):
    def test_replaces_matching_date_and_keeps_sorted_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "daily.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "report_date,value",
                        "2026-05-21,old",
                        "2026-05-22,keep",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            _upsert_daily_csv(
                csv_path,
                report_date="2026-05-21",
                row={"report_date": "2026-05-21", "value": "new"},
            )

            with csv_path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(
                rows,
                [
                    {"report_date": "2026-05-21", "value": "new"},
                    {"report_date": "2026-05-22", "value": "keep"},
                ],
            )


class RetentionPruneTests(unittest.TestCase):
    def test_deletes_only_old_notes_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            notes_dir = root / "notes"
            reports_dir = root / "reports"
            data_dir = root / "data"
            notes_dir.mkdir()
            reports_dir.mkdir()
            data_dir.mkdir()

            boundary = data_dir / "nys_boundary.geojson"
            boundary.write_text(
                '{"type":"Feature","geometry":{"type":"MultiPolygon","coordinates":[]}}',
                encoding="utf-8",
            )

            (notes_dir / "2026-01-01.md").write_text("old note\n", encoding="utf-8")
            (notes_dir / "2026-05-01.md").write_text("recent note\n", encoding="utf-8")
            (reports_dir / "2026-01-01").mkdir()
            (reports_dir / "2026-05-01").mkdir()
            (reports_dir / "latest").mkdir()
            (data_dir / "daily.csv").write_text("report_date\n2026-05-01\n", encoding="utf-8")

            cfg = DailyConfig(
                bbox="-79.8,40.45,-71.85,45.1",
                sample_size=10,
                stale_hours=12,
                run_date=date(2026, 5, 22),
                repo_root=root,
                ny_boundary_geojson=boundary,
                retention_days=91,
                _api_key="test-key",
            )

            result = prune_old_artifacts(cfg)

            self.assertEqual(result, {"deleted_notes": 1, "deleted_report_dirs": 1})
            self.assertFalse((notes_dir / "2026-01-01.md").exists())
            self.assertTrue((notes_dir / "2026-05-01.md").exists())
            self.assertFalse((reports_dir / "2026-01-01").exists())
            self.assertTrue((reports_dir / "2026-05-01").exists())
            self.assertTrue((reports_dir / "latest").exists())
            self.assertTrue((data_dir / "daily.csv").exists())


class WriteOutputsTests(unittest.TestCase):
    def test_records_fetch_warnings_in_note_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data"
            data_dir.mkdir()

            boundary = data_dir / "nys_boundary.geojson"
            boundary.write_text(
                "\n".join(
                    [
                        '{"type":"Feature","geometry":',
                        '{"type":"MultiPolygon","coordinates":',
                        '[[[[0.0,0.0],[10.0,0.0],[10.0,10.0],[0.0,10.0],[0.0,0.0]]]]}',
                        "}",
                    ]
                ),
                encoding="utf-8",
            )

            cfg = DailyConfig(
                bbox="-79.8,40.45,-71.85,45.1",
                sample_size=10,
                stale_hours=12,
                run_date=date(2026, 5, 22),
                repo_root=root,
                ny_boundary_geojson=boundary,
                retention_days=91,
                _api_key="test-key",
            )

            outputs = write_outputs(
                cfg,
                locations_latency_ms=42,
                ny_locations=[
                    {"id": 1, "coordinates": {"longitude": 1.0, "latitude": 1.0}},
                ],
                sampled_locations=[
                    {"id": 1, "coordinates": {"longitude": 1.0, "latitude": 1.0}},
                ],
                latest_elapsed_s=3.5,
                rows=[
                    {
                        "locationsId": 1,
                        "sensorsId": 7,
                        "datetime_utc": "2026-05-21T00:00:00Z",
                        "value": 12.3,
                        "latitude": 1.0,
                        "longitude": 1.0,
                        "stale": False,
                        "parameter_name": "pm25",
                        "units": "µg/m³",
                    }
                ],
                top_params=[("pm25", 1)],
                latest_errors=[(1, "timeout")],
            )

            note_text = outputs["note"].read_text(encoding="utf-8")
            with outputs["daily_csv"].open("r", newline="", encoding="utf-8") as f:
                csv_rows = list(csv.DictReader(f))

            self.assertIn("### Fetch warnings", note_text)
            self.assertIn("Location 1: timeout", note_text)
            self.assertEqual(csv_rows[0]["latest_error_count"], "1")


if __name__ == "__main__":
    unittest.main()
