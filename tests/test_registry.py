"""The owner registry: validation, CSV dry run versus apply, audit, and the licence match for drivers."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surveillance import registry as reg  # noqa: E402
from surveillance.store import Store  # noqa: E402

ADMIN = {"user_id": 1, "name": "Admin", "role": "admin", "address": "10.0.0.1"}
CSV = ("plate,owner_name,mobile,email,licence_no\n"
       "DHAKA METRO-GA 12-3456,Test Owner,01700000000,owner@example.test,DK0000001\n"
       "DHAKA METRO-KHA 99-1111,Other Owner,,,\n"
       "BAD,Nobody,,,\n"
       "DHAKA METRO-GA 12-3456,Duplicate,,,\n"
       "DHAKA METRO-HA 55-5555,Bad Mail,,not-an-email,\n")


class RegistryTests(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.store = Store(Path(d.name) / "s.db")
        self.addCleanup(self.store.db.close)
        reg.ensure_schema(self.store)

    def test_add_edit_delete_are_audited(self):
        row = reg.save(self.store, ADMIN, {"plate": "dhaka metro-ga 12-3456", "owner_name": "Test Owner",
                                           "email": "Owner@Example.test", "mobile": "01700000000"})
        self.assertEqual((row["plate_key"], row["email"]), ("DHAKAMETROGA123456", "owner@example.test"))
        reg.save(self.store, ADMIN, {"plate": "DHAKA METRO-GA 12-3456", "owner_name": "New Name"}, row["id"])
        self.assertEqual(reg.get(self.store, row["id"])["owner_name"], "New Name")
        reg.delete(self.store, ADMIN, row["id"])
        self.assertIsNone(reg.get(self.store, row["id"]))
        actions = [a["action"] for a in self.store.q("SELECT action FROM audit_log ORDER BY id")]
        self.assertEqual(actions, ["registry.add", "registry.edit", "registry.delete"])
        detail = " ".join(a["detail"] for a in self.store.q("SELECT detail FROM audit_log"))
        self.assertNotIn("owner@example.test", detail)       # the trail names fields, not personal data

    def test_duplicate_plate_and_bad_values_are_refused(self):
        reg.save(self.store, ADMIN, {"plate": "ABC-1234"})
        for bad in ({"plate": "abc 1234"}, {"plate": ""}, {"plate": "ABC-9999", "email": "x"},
                    {"plate": "ABC-9999", "mobile": "12"}):
            with self.assertRaises(reg.RegistryError):
                reg.save(self.store, ADMIN, bad)

    def test_dry_run_changes_nothing_and_apply_does_only_the_valid_lines(self):
        dry = reg.import_csv(self.store, ADMIN, CSV, apply=False)
        self.assertEqual(dry["counts"], {"add": 2, "update": 0, "same": 0, "error": 3})
        self.assertFalse(dry["applied"])
        self.assertEqual(reg.listing(self.store)[1], 0)
        done = reg.import_csv(self.store, ADMIN, CSV, apply=True)
        self.assertEqual(reg.listing(self.store)[1], 2)
        self.assertEqual(done["counts"]["add"], 2)
        again = reg.import_csv(self.store, ADMIN, CSV, apply=False)
        self.assertEqual(again["counts"]["same"], 2)
        self.assertTrue(any(a["action"] == "registry.import" for a in self.store.q("SELECT action FROM audit_log")))

    def test_csv_without_a_plate_column_is_refused(self):
        with self.assertRaises(reg.RegistryError):
            reg.import_csv(self.store, ADMIN, "name,email\nA,a@example.test\n")

    def test_licence_match_gives_a_driver_the_plate(self):
        reg.save(self.store, ADMIN, {"plate": "ABC-1234", "licence_no": "dk 000-0001"})
        uid = self.store.add_user("driver", "D", licence_no="DK0000001")
        self.assertEqual(reg.driver_plate_keys(self.store, {"user_id": uid, "licence_no": "DK0000001"}), {"ABC1234"})
        self.assertEqual(reg.driver_plate_keys(self.store, {"user_id": uid, "licence_no": ""}), set())
        self.assertEqual(reg.driver_plate_keys(self.store, {"user_id": uid, "licence_no": "DK9999999"}), set())


if __name__ == "__main__":
    unittest.main()
