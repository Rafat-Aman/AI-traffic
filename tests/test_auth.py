"""Accounts, sign-in and what each role may reach.

The security-relevant behaviour is pinned here rather than described, because the cost of getting it wrong
is somebody reading another person's violation history. In particular: a licence number must not be enough
to sign in, an unknown licence must look exactly like a known one, and a driver must never reach the
operator surface.
"""

import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surveillance import auth  # noqa: E402
from surveillance.store import Store  # noqa: E402


class HashTests(unittest.TestCase):
    def test_a_password_round_trips(self):
        h = auth.hash_secret("correct horse battery staple")
        self.assertTrue(auth.verify_secret("correct horse battery staple", h))
        self.assertFalse(auth.verify_secret("Correct horse battery staple", h))

    def test_the_password_is_not_in_the_hash(self):
        self.assertNotIn("hunter2", auth.hash_secret("hunter2"))

    def test_the_same_password_hashes_differently_each_time(self):
        self.assertNotEqual(auth.hash_secret("same"), auth.hash_secret("same"))

    def test_a_corrupt_or_missing_hash_is_a_refusal_not_a_crash(self):
        for bad in ("", None, "nonsense", "scrypt$x$y$z$q$r", "md5$1$2$3$4$5"):
            self.assertFalse(auth.verify_secret("x", bad))

    def test_an_empty_secret_never_verifies(self):
        self.assertFalse(auth.verify_secret("", auth.hash_secret("x")))


class TokenTests(unittest.TestCase):
    def test_session_tokens_are_unique_and_long(self):
        seen = {auth.new_token() for _ in range(200)}
        self.assertEqual(len(seen), 200)
        self.assertGreaterEqual(len(seen.pop()), 32)

    def test_the_database_stores_a_fingerprint_not_the_token(self):
        t = auth.new_token()
        fp = auth.token_fingerprint(t)
        self.assertNotIn(t, fp)
        self.assertEqual(fp, auth.token_fingerprint(t))

    def test_codes_are_the_right_shape(self):
        for _ in range(50):
            c = auth.new_code()
            self.assertEqual(len(c), auth.OTP_DIGITS)
            self.assertTrue(c.isdigit())


class NormaliseTests(unittest.TestCase):
    def test_licences_compare_without_punctuation_or_case(self):
        self.assertEqual(auth.normalise_licence("dk-123 4567"), auth.normalise_licence("DK1234567"))

    def test_plates_compare_the_way_the_store_searches_them(self):
        self.assertEqual(auth.normalise_plate("DHAKA METRO-GA 12-3456"), "DHAKAMETROGA123456")

    def test_a_masked_phone_keeps_only_the_last_digits(self):
        masked = auth.mask_phone("+8801712345678")
        self.assertTrue(masked.endswith("678"))
        self.assertNotIn("1712345", masked)


class RoleTests(unittest.TestCase):
    def test_admin_covers_controller_but_not_the_other_way(self):
        self.assertTrue(auth.can(auth.ADMIN, auth.CONTROLLER))
        self.assertTrue(auth.can(auth.ADMIN, auth.ADMIN))
        self.assertFalse(auth.can(auth.CONTROLLER, auth.ADMIN))

    def test_a_driver_is_only_ever_a_driver(self):
        self.assertTrue(auth.can(auth.DRIVER, auth.DRIVER))
        self.assertFalse(auth.can(auth.DRIVER, auth.CONTROLLER))
        self.assertFalse(auth.can(auth.DRIVER, auth.ADMIN))

    def test_nobody_is_implicitly_a_driver(self):
        """Staff must not fall through into a driver's portal: it would need an owner to make sense of."""
        self.assertFalse(auth.can(auth.ADMIN, auth.DRIVER))
        self.assertFalse(auth.can(auth.CONTROLLER, auth.DRIVER))


class RateLimitTests(unittest.TestCase):
    def test_it_allows_the_quota_then_refuses(self):
        r = auth.RateLimiter(limit=3, window_sec=60)
        self.assertEqual([r.check("k")[0] for _ in range(4)], [True, True, True, False])

    def test_keys_are_counted_separately(self):
        r = auth.RateLimiter(limit=1, window_sec=60)
        self.assertTrue(r.check("a")[0])
        self.assertTrue(r.check("b")[0])
        self.assertFalse(r.check("a")[0])

    def test_it_says_how_long_to_wait(self):
        r = auth.RateLimiter(limit=1, window_sec=30)
        r.check("k")
        ok, wait = r.check("k")
        self.assertFalse(ok)
        self.assertTrue(0 < wait <= 31)

    def test_the_window_slides(self):
        r = auth.RateLimiter(limit=1, window_sec=0.3)
        r.check("k")
        self.assertFalse(r.check("k")[0])
        time.sleep(0.35)
        self.assertTrue(r.check("k")[0])

    def test_clearing_forgets_a_key(self):
        r = auth.RateLimiter(limit=1, window_sec=60)
        r.check("k")
        r.clear("k")
        self.assertTrue(r.check("k")[0])


class EnumerationTests(unittest.TestCase):
    """A licence number is public enough that this endpoint must not answer "does this person hold one".

    Caught in testing: the reply carried a masked phone number when the licence existed and omitted it
    when it did not, which answered the question precisely."""

    def setUp(self):
        import surveillance.server as srv
        self.srv = srv

    def test_the_request_code_reply_has_no_field_that_depends_on_existence(self):
        import inspect
        src = inspect.getsource(self.srv.request_code)
        body = src[src.index("out = {"):]
        self.assertNotIn("phone_hint", body,
                         "a field only present for known licences tells an attacker the licence exists")

    def test_the_wording_is_conditional(self):
        import inspect
        self.assertIn("If that licence is registered", inspect.getsource(self.srv.request_code))

    def test_a_failed_staff_login_does_not_say_which_half_was_wrong(self):
        import inspect
        src = inspect.getsource(self.srv.staff_login)
        self.assertIn("Wrong username or password", src)
        self.assertNotIn("No such user", src)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.dir.name) / "s.db")

    def tearDown(self):
        self.dir.cleanup()

    def driver(self, licence="DK1234567", plate="DHAKAMETROGA123456", verified=1):
        uid = self.store.add_user(auth.DRIVER, "Karim", licence_no=licence, phone="+8801712345678")
        self.store.add_vehicle(plate, uid, verified=verified)
        return uid

    def confirmed(self, plate_en, status="confirmed"):
        cid = self.store.add_camera("c", "file", "v.mp4")
        admin = self.store.user_by_username("a") or {"id": self.store.add_user(
            auth.ADMIN, "Admin", username="a", password_hash="x")}
        rec = {"code": "V1", "violation": "Bus stopped", "track_id": 1, "class": "Bus",
               "class_confidence": 0.9, "video_time_s": 1.0, "started_s": 0.0, "wall_time": "",
               "zone": "z", "areas": "", "kph": 0.0, "details": {}}
        vid = self.store.add_violation(cid, None, None, rec, "s.jpg", "c.jpg", "c.mp4")
        self.store.set_plate(vid, "read", "x", plate_en, 0.9, None)
        self.store.review(vid, status, "", by={"user_id": admin["id"], "name": "Admin"})
        return vid

    def test_a_driver_sees_only_their_own_plates(self):
        uid = self.driver()
        mine = self.confirmed("DHAKA METRO-GA 12-3456")
        self.confirmed("DHAKA METRO-KA 77-7777")
        keys = [v["plate_key"] for v in self.store.vehicles_of(uid, verified_only=True)]
        rows, total = self.store.violations_for_plates(keys)
        self.assertEqual(total, 1)
        self.assertEqual(rows[0]["id"], mine)

    def test_plate_matching_ignores_spacing_and_case(self):
        uid = self.driver(plate=auth.normalise_plate("dhaka metro ga 123456"))
        self.confirmed("DHAKA METRO-GA 12-3456")
        keys = [v["plate_key"] for v in self.store.vehicles_of(uid, verified_only=True)]
        self.assertEqual(self.store.violations_for_plates(keys)[1], 1)

    def test_an_unverified_vehicle_shows_nothing(self):
        uid = self.driver(verified=0)
        self.confirmed("DHAKA METRO-GA 12-3456")
        keys = [v["plate_key"] for v in self.store.vehicles_of(uid, verified_only=True)]
        self.assertEqual(keys, [])
        self.assertEqual(self.store.violations_for_plates(keys)[1], 0)

    def test_cases_still_in_the_queue_are_not_shown(self):
        """Nobody has decided a pending case, so telling its keeper about it would be telling them they
        are liable for something no human has looked at."""
        uid = self.driver()
        self.confirmed("DHAKA METRO-GA 12-3456", status="pending")
        keys = [v["plate_key"] for v in self.store.vehicles_of(uid, verified_only=True)]
        self.assertEqual(self.store.violations_for_plates(keys)[1], 0)

    def test_no_plates_means_no_query_at_all(self):
        self.assertEqual(self.store.violations_for_plates([]), ([], 0))

    def test_a_review_records_who_made_it(self):
        vid = self.confirmed("DHAKA METRO-GA 12-3456")
        uid = self.store.add_user(auth.CONTROLLER, "Sgt Rahman", username="rahman",
                                  password_hash=auth.hash_secret("x"))
        self.store.review(vid, "rejected", "plate misread",
                          by={"user_id": uid, "name": "Sgt Rahman", "role": "controller"})
        row = self.store.violation(vid)
        self.assertEqual(row["reviewed_by"], uid)
        self.assertEqual(row["reviewed_by_name"], "Sgt Rahman")

    def test_a_licence_is_unique(self):
        self.driver()
        import sqlite3
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.add_user(auth.DRIVER, "Someone else", licence_no="DK1234567", phone="+88017")

    def test_sessions_resolve_and_expire(self):
        uid = self.driver()
        token = auth.new_token()
        self.store.add_session(auth.token_fingerprint(token), uid, auth.DRIVER, time.time() + 60)
        self.assertIsNotNone(self.store.session(auth.token_fingerprint(token)))
        self.store.purge_sessions(time.time() + 120)
        self.assertIsNone(self.store.session(auth.token_fingerprint(token)))

    def test_disabling_an_account_can_end_its_sessions(self):
        uid = self.driver()
        t = auth.new_token()
        self.store.add_session(auth.token_fingerprint(t), uid, auth.DRIVER, time.time() + 600)
        self.store.drop_sessions_of(uid)
        self.assertIsNone(self.store.session(auth.token_fingerprint(t)))

    def test_a_login_code_is_single_use(self):
        uid = self.driver()
        self.store.add_login_code(uid, auth.hash_secret("123456"), time.time() + 300)
        row = self.store.live_login_code(uid, time.time())
        self.assertIsNotNone(row)
        self.store.spend_login_code(row["id"])
        self.assertIsNone(self.store.live_login_code(uid, time.time()))

    def test_asking_for_a_new_code_retires_the_old_one(self):
        uid = self.driver()
        self.store.add_login_code(uid, auth.hash_secret("111111"), time.time() + 300)
        first = self.store.live_login_code(uid, time.time())["id"]
        self.store.add_login_code(uid, auth.hash_secret("222222"), time.time() + 300)
        self.assertNotEqual(self.store.live_login_code(uid, time.time())["id"], first)

    def test_an_expired_code_is_not_live(self):
        uid = self.driver()
        self.store.add_login_code(uid, auth.hash_secret("123456"), time.time() - 1)
        self.assertIsNone(self.store.live_login_code(uid, time.time()))

    def test_the_audit_trail_records_who_and_what(self):
        self.store.audit({"user_id": 7, "name": "Sgt Rahman", "role": "controller"},
                         "violation.confirmed", "violation", 3, "looked right")
        row = self.store.audit_trail("violation", 3)[0]
        self.assertEqual((row["who"], row["role"], row["action"]),
                         ("Sgt Rahman", "controller", "violation.confirmed"))

    def test_attribution_survives_the_reviewer_being_removed(self):
        """An account can be closed; who decided a case cannot become unknown because of it."""
        uid = self.store.add_user(auth.CONTROLLER, "Sgt Rahman", username="r", password_hash="x")
        vid = self.confirmed("DHAKA METRO-GA 12-3456")
        self.store.review(vid, "confirmed", "", by={"user_id": uid, "name": "Sgt Rahman"})
        self.store.x("DELETE FROM users WHERE id=?", (uid,))
        self.assertEqual(self.store.violation(vid)["reviewed_by_name"], "Sgt Rahman")

    def test_the_audit_trail_survives_the_account_being_removed(self):
        uid = self.store.add_user(auth.CONTROLLER, "Gone", username="gone", password_hash="x")
        self.store.audit({"user_id": uid, "name": "Gone", "role": "controller"}, "violation.confirmed",
                         "violation", 1)
        self.store.x("DELETE FROM users WHERE id=?", (uid,))
        self.assertEqual(self.store.audit_trail("violation", 1)[0]["who"], "Gone")


if __name__ == "__main__":
    unittest.main(verbosity=2)
