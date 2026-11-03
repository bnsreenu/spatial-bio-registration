"""Review status rules and batch_summary rows (no Qt)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import session  # noqa: E402
from app import review_status as rs  # noqa: E402


def result(sid, confidence="high", status="ok", inliers=(120, 300), ratios=(0.8, 0.9), diffs=(None, 0.07)):
    return {"sample_id": sid, "status": status, "confidence": confidence, "reference_modality": "xenium",
            "warnings": ["line one\nline two", "second"],
            "diagnostics": [{"step": f"s{i}", "n_inliers": n, "inlier_ratio": r, "second_opinion_diff_mean_um": d}
                            for i, (n, r, d) in enumerate(zip(inliers, ratios, diffs))]}


class TestReviewStatus(unittest.TestCase):
    def test_labels_and_defaults(self):
        high, low, failed = result("a"), result("b", "low"), result("c", None, "failed")
        item = rs.default_entry()
        self.assertEqual(rs.label(high, item), "unreviewed")
        self.assertEqual(rs.label(low, item), "needs attention")
        self.assertEqual(rs.label(failed, item), "needs attention")
        review = {}
        rs.set_status(review, "b", rs.ACCEPTED)
        self.assertEqual(rs.label(low, rs.entry(review, "b")), "accepted")
        rs.set_status(review, "b", rs.ACCEPTED, bulk=True)
        self.assertEqual(rs.label(low, rs.entry(review, "b")), "accepted (bulk)")
        rs.set_status(review, "b", rs.REJECTED, bulk=True)
        self.assertEqual(rs.label(low, rs.entry(review, "b")), "rejected")
        self.assertFalse(review["b"]["bulk"])

    def test_bulk_accepts_only_unreviewed_high(self):
        results = {"a": result("a"), "b": result("b", "low"), "c": result("c"), "d": result("d", "medium"),
                   "e": result("e", None, "failed")}
        review = {}
        rs.set_status(review, "c", rs.REJECTED)
        self.assertEqual(rs.bulk_candidates(results, review), ["a"])
        self.assertEqual(rs.accept_bulk(results, review), ["a"])
        self.assertEqual(review["a"], {"status": "accepted", "bulk": True})
        self.assertEqual(review["c"]["status"], "rejected")
        self.assertEqual(rs.accept_bulk(results, review), [])

    def test_to_write(self):
        results = {"a": result("a"), "b": result("b")}
        review = {}
        rs.set_status(review, "a", rs.ACCEPTED)
        self.assertEqual(rs.to_write(results, review, False), ["a"])
        self.assertEqual(rs.to_write(results, review, True), ["a", "b"])

    def test_clean_session_value(self):
        self.assertEqual(rs.clean(None), rs.default_entry())
        self.assertEqual(rs.clean({"status": "bogus"}), rs.default_entry())
        self.assertEqual(rs.clean({"status": "accepted", "bulk": True}), {"status": "accepted", "bulk": True})
        self.assertEqual(rs.clean({"status": "rejected", "bulk": True}), {"status": "rejected", "bulk": False})


class TestSummaryRow(unittest.TestCase):
    def test_row_values(self):
        row = session.summary_row(result("a"), "xenium", "accepted (bulk)", "2026-01-01T00:00:00", "v1")
        self.assertEqual(list(row), session.SUMMARY_COLUMNS)
        self.assertEqual((row["inliers_weakest_step"], row["inlier_ratio_weakest_step"],
                          row["sift_diff_um_worst_step"]), (120, 0.8, 0.07))
        self.assertEqual(row["warnings"], "line one line two | second")
        self.assertEqual((row["review_status"], row["export_time"], row["tool_version"]),
                         ("accepted (bulk)", "2026-01-01T00:00:00", "v1"))

    def test_blank_when_missing(self):
        row = session.summary_row(result("a", None, "failed", (), (), ()), None, "needs attention", "", "v1")
        self.assertEqual((row["inliers_weakest_step"], row["inlier_ratio_weakest_step"],
                          row["sift_diff_um_worst_step"]), ("", "", ""))
        self.assertEqual(row["confidence"], "failed")
        self.assertEqual(row["reference"], "xenium")

    def test_session_round_trip_keeps_review(self):
        import tempfile
        res = result("a")
        res["matrices"] = {}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a_session.json"
            session.save_session(path, {}, {}, res, review={"status": "accepted", "bulk": True})
            self.assertEqual(session.load_session(path)["review"], {"status": "accepted", "bulk": True})
            session.save_session(path, {}, {}, res)
            self.assertIsNone(session.load_session(path)["review"])


if __name__ == "__main__":
    unittest.main()
