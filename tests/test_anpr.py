"""Number-plate parsing and multi-frame voting (no models needed)."""

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surveillance.anpr import PlateRead, parse_plate, vote  # noqa: E402


def plate_dets(number="123456", city="Dhaka", letter="go", conf=0.9):
    """Character boxes of a 300x160 two-line plate: 'Dhaka metro-<letter>' over '12-3456'."""
    d = [(city, conf, 20, 15, 110, 60), ("Dha", conf, 20, 15, 65, 60), ("ka", conf, 65, 15, 110, 60),
         ("meTrO", conf, 120, 15, 230, 60), ("me", conf, 120, 15, 170, 60), ("TrO", conf, 170, 15, 230, 60),
         (letter, conf, 245, 15, 280, 60)]
    xs = [20, 60, 120, 160, 200, 240]
    d += [(ch, conf, x, 85, x + 35, 145) for ch, x in zip(number, xs)]
    return d


class ParseTests(unittest.TestCase):
    def test_full_plate(self):
        r = parse_plate(plate_dets(), 300, 160, plate_px=150)
        self.assertTrue(r.complete)
        self.assertEqual(r.text("bn"), "ঢাকা মেট্রো-গ ১২-৩৪৫৬")
        self.assertEqual(r.text("en"), "DHAKA METRO-GA 12-3456")

    def test_ka_inside_dhaka_is_not_the_class_letter(self):
        r = parse_plate(plate_dets(letter="ka"), 300, 160)
        self.assertEqual(r.letter, "ka")  # the real class letter (right of metro) ...
        d = [x for x in plate_dets() if x[0] != "go"]  # ... and with no letter there, "ka" of Dhaka is not used
        self.assertIsNone(parse_plate(d, 300, 160).letter)

    def test_duplicate_digit_boxes(self):
        d = plate_dets() + [("8", 0.5, 22, 86, 56, 144)]  # weaker second guess on the first digit
        self.assertEqual(parse_plate(d, 300, 160).digits, list("123456"))

    def test_missing_digit(self):
        d = [x for x in plate_dets() if x[0] != "4"]
        self.assertFalse(parse_plate(d, 300, 160).number_ok)


class VoteTests(unittest.TestCase):
    def read(self, number="123456", px=150, conf=0.9):
        return parse_plate(plate_dets(number, conf=conf), 300, 160, plate_px=px)

    def test_agreeing_reads(self):
        r = vote([self.read(), self.read(), self.read(), self.read("123486", conf=0.6)])
        self.assertEqual(r.status, "read")
        self.assertEqual(r.text_en, "DHAKA METRO-GA 12-3456")
        self.assertEqual((r.readings, r.agreeing), (4, 3))

    def test_two_agreeing_reads_are_not_enough(self):
        self.assertEqual(vote([self.read(), self.read(), self.read("654321")]).status, "unreadable")

    def test_low_digit_confidence_does_not_count(self):
        self.assertEqual(vote([self.read(conf=0.65)] * 4).status, "unreadable")

    def test_single_unclear_read_is_not_reported(self):
        self.assertEqual(vote([self.read(conf=0.5)]).status, "unreadable")

    def test_too_small_plates_are_not_read(self):
        self.assertEqual(vote([self.read(px=90), self.read(px=100), self.read(px=105)]).status, "unreadable")

    def test_disagreeing_reads(self):
        self.assertEqual(vote([self.read("123456"), self.read("654321")]).status, "unreadable")

    def test_nothing(self):
        self.assertEqual(vote([]).status, "unreadable")

    def test_number_only(self):
        reads = [parse_plate([x for x in plate_dets() if x[0].isdigit()], 300, 160, 150) for _ in range(3)]
        r = vote(reads)
        self.assertEqual(r.status, "partial")
        self.assertIn("12-3456", r.text_en)


class StoreMigrationTests(unittest.TestCase):
    def test_old_database_gets_plate_columns(self):
        from surveillance.store import Store
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "old.db"
            con = sqlite3.connect(db)
            con.execute("CREATE TABLE violations (id INTEGER PRIMARY KEY, camera_id INTEGER, code TEXT, title TEXT,"
                        " details TEXT, status TEXT DEFAULT 'pending', created_at TEXT)")
            con.commit()
            con.close()
            s = Store(db)
            cols = {r[1] for r in s.db.execute("PRAGMA table_info(violations)")}
            s.db.close()
            self.assertTrue({"plate_status", "plate_text", "plate_text_en", "plate_conf", "plate_img"} <= cols)


if __name__ == "__main__":
    unittest.main()
