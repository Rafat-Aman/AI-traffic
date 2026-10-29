"""E-prosecution: what may be cited, the evidence seal, signed media links, and who can reach what.

These pin the behaviour that matters when a citation is disputed or a link is passed around: a speed case
must never become a citation, a sealed file that changes by one byte must be found, a signed link must stop
working when it expires or is edited, and nothing a URL says may reach a file outside data/evidence.

Importing the dashboard server opens its database, so the server is pointed at a scratch folder before
anything imports it; no test here touches data/surveillance.db.
"""

import atexit
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if "surveillance.server" not in sys.modules:
    _SCRATCH = tempfile.mkdtemp(prefix="citation-server-")
    os.environ["TRAFFIC_DATA_DIR"] = _SCRATCH
    atexit.register(shutil.rmtree, _SCRATCH, ignore_errors=True)

from surveillance import auth  # noqa: E402
from surveillance import citation as cit  # noqa: E402
from surveillance.store import Store  # noqa: E402

SECRET = b"unit-test-secret-not-used-anywhere-else"


def make_case(root, **over):
    """A confirmed case with a real file for every piece of evidence, under root/evidence. Returns the case row."""
    folder = Path(root) / "evidence" / "cam1" / "run2"
    folder.mkdir(parents=True, exist_ok=True)
    names = {"snapshot": "V3_t4_12.0s.jpg", "crop": "V3_t4_12.0s_crop.jpg", "clip": "V3_t4_12.0s.mp4",
             "plate_img": "plate_7.jpg", "stopped": "V3_t4_12.0s_stopped.jpg"}
    for key, name in names.items():
        (folder / name).write_bytes((key.encode() + b"-bytes-") * 500)
    rel = {k: f"evidence/cam1/run2/{n}" for k, n in names.items()}
    v = {"id": 7, "camera_id": 1, "code": "V3", "title": "Kerb parking", "status": "confirmed",
         "plate_status": "read", "plate_text": "ঢাকা মেট্রো-গ ১২-৩৪৫৬", "plate_text_en": "DHAKA METRO-GA 12-3456",
         "wall_time": "2026-10-25 09:30:12", "video_time_s": 12.0, "snapshot": rel["snapshot"],
         "crop": rel["crop"], "clip": rel["clip"], "plate_img": rel["plate_img"],
         "evidence": [{"label": "stopped", "time_s": 5.0, "wall_time": "", "path": rel["stopped"]}]}
    v.update(over)
    return v


def sealed(v, root):
    text, bundle = cit.build_manifest(v, root)
    return {"citation_no": "DMP-2026-000001", "manifest": text, "bundle_sha256": bundle}


class Scratch(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        self.addCleanup(self._dir.cleanup)


class AdvisoryPolicyTests(unittest.TestCase):
    def test_speed_is_advisory_unless_the_environment_says_otherwise(self):
        """The report: speed "never supports a speeding citation". The default has to be that position."""
        self.assertEqual(cit.advisory_codes({}), frozenset({"V2"}))

    def test_the_environment_replaces_the_list(self):
        self.assertEqual(cit.advisory_codes({"ADVISORY_CODES": "v2, V6 ,"}), frozenset({"V2", "V6"}))

    def test_an_empty_setting_means_nothing_is_advisory(self):
        self.assertEqual(cit.advisory_codes({"ADVISORY_CODES": ""}), frozenset())

    def test_every_other_rule_stays_citable(self):
        for code in ("V1", "V3", "V4", "V5", "V6", "V7", "V8"):
            self.assertFalse(cit.is_advisory(code, cit.advisory_codes({})), code)

    def test_matching_ignores_case_and_stray_spaces(self):
        self.assertTrue(cit.is_advisory(" v2 ", frozenset({"V2"})))

    def test_the_running_default_matches_the_report(self):
        if "ADVISORY_CODES" not in os.environ:
            self.assertTrue(cit.is_advisory("V2"))


class FineTests(unittest.TestCase):
    def test_no_amount_is_invented(self):
        """The Road Transport Act 2018 schedule is the authority's to supply; every code starts unset."""
        self.assertEqual(cit.fine_schedule({}), {})
        for code in ("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"):
            self.assertIsNone(cit.fine_for(code, {}))

    def test_an_unset_fine_is_shown_as_the_authoritys_to_set(self):
        self.assertEqual(cit.fine_text(None), "set by the authority")
        self.assertEqual(cit.fine_text(1500), "BDT 1,500")

    def test_configured_amounts_are_used_for_their_code_only(self):
        sched = cit.fine_schedule({"FINE_SCHEDULE_BDT": '{"v3": 1000, "V7": 500}'})
        self.assertEqual(cit.fine_for("V3", sched), 1000)
        self.assertEqual(cit.fine_for("V7", sched), 500)
        self.assertIsNone(cit.fine_for("V1", sched))

    def test_entries_that_are_not_whole_positive_taka_are_ignored_and_reported(self):
        said = []
        sched = cit.fine_schedule({"FINE_SCHEDULE_BDT":
                                   '{"V1": "1000", "V3": 12.5, "V4": -5, "V5": 0, "V6": true, "V7": 700}'},
                                  log=said.append)
        self.assertEqual(sched, {"V7": 700})
        self.assertEqual(len(said), 5)

    def test_malformed_json_sets_nothing(self):
        said = []
        self.assertEqual(cit.fine_schedule({"FINE_SCHEDULE_BDT": "{nope"}, log=said.append), {})
        self.assertEqual(cit.fine_schedule({"FINE_SCHEDULE_BDT": "[1, 2]"}, log=said.append), {})
        self.assertEqual(len(said), 2)

    def test_the_prefix_falls_back_when_it_is_not_letters(self):
        self.assertEqual(cit.citation_prefix({}), "DMP")
        self.assertEqual(cit.citation_prefix({"CITATION_PREFIX": "cmp"}), "CMP")
        self.assertEqual(cit.citation_prefix({"CITATION_PREFIX": "../x"}), "DMP")


class PathSafetyTests(Scratch):
    def test_ordinary_relative_paths_pass(self):
        self.assertEqual(cit.clean_rel("evidence/cam1/run2/V3_t4_12.0s.jpg"), "evidence/cam1/run2/V3_t4_12.0s.jpg")

    def test_every_way_out_of_the_folder_is_refused(self):
        for bad in ("../x", "a/../../x", "a/..", "/etc/passwd", "a//b", "a/./b", "a\\b", "a\0b", "", None, 5, ".."):
            self.assertIsNone(cit.clean_rel(bad), repr(bad))

    def test_only_files_under_evidence_are_reachable_as_evidence(self):
        make_case(self.root)
        (self.root / "autozone").mkdir()
        (self.root / "autozone" / "a.jpg").write_bytes(b"x")
        (self.root / "surveillance.db").write_bytes(b"x")
        self.assertIsNotNone(cit.evidence_file(self.root, "evidence/cam1/run2/V3_t4_12.0s.jpg"))
        for other in ("autozone/a.jpg", "surveillance.db", "evidence", "evidence/", "evidence/../surveillance.db",
                      "evidence/cam1/run2/../../../surveillance.db", "evidence/cam1/run2/missing.jpg"):
            self.assertIsNone(cit.evidence_file(self.root, other), other)

    def test_a_folder_is_not_a_file(self):
        make_case(self.root)
        self.assertIsNone(cit.evidence_file(self.root, "evidence/cam1/run2"))

    def test_a_symlink_that_leaves_the_folder_is_refused(self):
        make_case(self.root)
        outside = self.root / "outside.txt"
        outside.write_text("secret")
        try:
            (self.root / "evidence" / "cam1" / "link.txt").symlink_to(outside)
        except OSError:
            self.skipTest("symlinks are not available here")
        self.assertIsNone(cit.evidence_file(self.root, "evidence/cam1/link.txt"))


class SealTests(Scratch):
    def test_the_manifest_hash_is_stable(self):
        v = make_case(self.root)
        self.assertEqual(cit.build_manifest(v, self.root), cit.build_manifest(dict(v), self.root))

    def test_the_order_the_files_are_listed_in_does_not_matter(self):
        v = make_case(self.root)
        flipped = dict(v, evidence=list(reversed(v["evidence"])))
        self.assertEqual(cit.build_manifest(v, self.root)[1], cit.build_manifest(flipped, self.root)[1])

    def test_every_evidence_file_is_sealed_once_and_listed_in_path_order(self):
        import json
        m = json.loads(cit.build_manifest(make_case(self.root), self.root)[0])
        paths = [f["path"] for f in m["files"]]
        self.assertEqual(paths, sorted(paths))
        self.assertEqual({f["role"] for f in m["files"]},
                         {"snapshot", "crop", "clip", "plate", "evidence:stopped"})
        self.assertEqual(len(paths), 5)

    def test_the_manifest_carries_the_facts_of_the_case(self):
        import json
        m = json.loads(cit.build_manifest(make_case(self.root), self.root)[0])
        self.assertEqual((m["violation_id"], m["code"], m["plate"], m["wall_time"], m["camera_id"]),
                         (7, "V3", "DHAKA METRO-GA 12-3456", "2026-10-25 09:30:12", 1))

    def test_the_canonical_text_is_ascii_even_for_a_bangla_plate(self):
        """So the hash cannot depend on how a tool normalises or re-encodes Bangla text."""
        import json
        text = cit.build_manifest(make_case(self.root), self.root)[0]
        self.assertTrue(text.isascii())
        self.assertIn("\\u09a2", text, "the plate is there, escaped")
        # canonical = canonicalising it again changes nothing, so an independent tool that re-serialises
        # the parsed manifest the same way gets the same bytes and the same hash
        self.assertEqual(text, cit.canonical(json.loads(text)))

    def test_the_bundle_hash_is_the_sha256_of_the_manifest_text(self):
        import hashlib
        text, bundle = cit.build_manifest(make_case(self.root), self.root)
        self.assertEqual(bundle, hashlib.sha256(text.encode()).hexdigest())

    def test_an_untouched_bundle_verifies(self):
        v = make_case(self.root)
        r = cit.verify_seal(sealed(v, self.root), v, self.root)
        self.assertTrue(r["ok"])
        self.assertEqual(r["counts"], {"match": 5, "mismatch": 0, "missing": 0})

    def test_one_modified_byte_is_found_and_the_file_is_named(self):
        v = make_case(self.root)
        c = sealed(v, self.root)
        clip = self.root / v["clip"]
        data = bytearray(clip.read_bytes())
        data[100] ^= 0x01
        clip.write_bytes(bytes(data))
        r = cit.verify_seal(c, v, self.root)
        self.assertFalse(r["ok"])
        self.assertEqual({f["path"]: f["status"] for f in r["files"] if f["status"] != "match"},
                         {v["clip"]: "mismatch"})
        self.assertEqual(r["counts"], {"match": 4, "mismatch": 1, "missing": 0})

    def test_a_missing_file_is_reported_missing_not_as_a_match(self):
        v = make_case(self.root)
        c = sealed(v, self.root)
        (self.root / v["crop"]).unlink()
        r = cit.verify_seal(c, v, self.root)
        self.assertFalse(r["ok"])
        self.assertEqual([f["status"] for f in r["files"] if f["path"] == v["crop"]], ["missing"])

    def test_an_edited_manifest_is_caught_by_the_bundle_hash(self):
        v = make_case(self.root)
        c = sealed(v, self.root)
        c["manifest"] = c["manifest"].replace("V3", "V1", 1)
        r = cit.verify_seal(c, v, self.root)
        self.assertEqual(r["bundle"]["status"], "mismatch")
        self.assertFalse(r["ok"])

    def test_a_manifest_and_hash_edited_together_still_disagree_with_the_files(self):
        """Re-sealing the text is not enough: the files on the disk have to match what it lists."""
        v = make_case(self.root)
        c = sealed(v, self.root)
        (self.root / v["snapshot"]).write_bytes(b"replaced")
        r = cit.verify_seal(c, v, self.root)
        self.assertEqual(r["bundle"]["status"], "match")
        self.assertFalse(r["ok"])

    def test_a_changed_case_record_is_reported(self):
        v = make_case(self.root)
        c = sealed(v, self.root)
        r = cit.verify_seal(c, dict(v, plate_text_en="DHAKA METRO-KA 99-9999"), self.root)
        self.assertEqual(r["case"]["plate"], "mismatch")
        self.assertEqual(r["case"]["code"], "match")
        self.assertFalse(r["ok"])

    def test_a_deleted_case_is_reported(self):
        v = make_case(self.root)
        r = cit.verify_seal(sealed(v, self.root), None, self.root)
        self.assertFalse(r["ok"])
        self.assertTrue(all(s == "missing" for s in r["case"].values()))

    def test_a_manifest_that_is_not_json_does_not_verify_or_crash(self):
        v = make_case(self.root)
        r = cit.verify_seal({"citation_no": "X", "manifest": "{broken", "bundle_sha256": "0" * 64}, v, self.root)
        self.assertFalse(r["ok"])

    def test_sealing_refuses_a_case_whose_file_is_gone(self):
        v = make_case(self.root)
        (self.root / v["plate_img"]).unlink()
        with self.assertRaisesRegex(cit.Refused, "plate_7.jpg"):
            cit.build_manifest(v, self.root)

    def test_a_stored_path_cannot_point_the_seal_outside_evidence(self):
        v = make_case(self.root, snapshot="../surveillance.db")
        (self.root / "surveillance.db").write_bytes(b"x")
        with self.assertRaises(cit.Refused):
            cit.build_manifest(v, self.root)

    def test_the_summary_says_what_is_wrong(self):
        v = make_case(self.root)
        c = sealed(v, self.root)
        (self.root / v["crop"]).unlink()
        (self.root / v["clip"]).write_bytes(b"other")
        said = cit.verify_summary(cit.verify_seal(c, v, self.root))
        self.assertIn("1 mismatch", said)
        self.assertIn("1 missing", said)


class EligibilityTests(Scratch):
    def why(self, **over):
        return cit.blockers(make_case(self.root, **over), self.root, advisory=frozenset({"V2"}))

    def test_a_confirmed_readable_case_may_be_cited(self):
        self.assertEqual(self.why(), [])

    def test_a_speed_case_is_never_citable(self):
        self.assertTrue(any("advisory" in w for w in self.why(code="V2")))

    def test_only_confirmed_cases_may_be_cited(self):
        for status in ("pending", "rejected"):
            self.assertTrue(any(status in w for w in self.why(status=status)), status)

    def test_a_citation_needs_a_plate_that_was_read(self):
        self.assertEqual(self.why(plate_status="partial"), [])
        for status in ("", "pending", "unreadable"):
            self.assertTrue(any("plate" in w for w in self.why(plate_status=status)), status)

    def test_a_clip_that_is_still_being_written_blocks_it(self):
        v = make_case(self.root)
        final = self.root / v["clip"]
        final.rename(final.with_suffix(".part.mp4"))
        self.assertTrue(any("still being encoded" in w for w in cit.blockers(v, self.root, frozenset())))

    def test_a_clip_that_is_not_there_blocks_it(self):
        v = make_case(self.root)
        (self.root / v["clip"]).unlink()
        self.assertTrue(any("not on the disk" in w for w in cit.blockers(v, self.root, frozenset())))

    def test_a_case_with_no_clip_at_all_is_blocked(self):
        self.assertTrue(any("no video clip" in w for w in self.why(clip=None)))

    def test_every_reason_is_given_not_just_the_first(self):
        self.assertGreaterEqual(len(self.why(code="V2", status="pending", plate_status="unreadable")), 3)


class SignedLinkTests(unittest.TestCase):
    PATH = "evidence/cam1/run2/V3_t4_12.0s.jpg"

    def test_a_fresh_link_is_valid(self):
        self.assertEqual(cit.check_token(cit.sign(self.PATH, SECRET), self.PATH, SECRET), "ok")

    def test_it_expires_after_its_lifetime_and_not_before(self):
        tok = cit.sign(self.PATH, SECRET, ttl=60, now=1000)
        self.assertEqual(cit.check_token(tok, self.PATH, SECRET, now=1060), "ok")
        self.assertEqual(cit.check_token(tok, self.PATH, SECRET, now=1061), "expired")

    def test_the_default_lifetime_is_an_hour(self):
        self.assertEqual(cit._link_ttl({}), 3600)
        tok = cit.sign(self.PATH, SECRET, now=5000)
        self.assertEqual(int(tok.split(".")[0]), 5000 + cit.LINK_TTL_SEC)

    def test_the_lifetime_can_be_set_but_not_to_nothing_or_forever(self):
        self.assertEqual(cit._link_ttl({"MEDIA_LINK_TTL_SEC": "120"}), 120)
        self.assertEqual(cit._link_ttl({"MEDIA_LINK_TTL_SEC": "5"}), 60)
        self.assertEqual(cit._link_ttl({"MEDIA_LINK_TTL_SEC": "999999999"}), 7 * 24 * 3600)
        self.assertEqual(cit._link_ttl({"MEDIA_LINK_TTL_SEC": "soon"}), 3600)

    def test_a_link_for_one_file_does_not_open_another(self):
        tok = cit.sign(self.PATH, SECRET)
        self.assertEqual(cit.check_token(tok, "evidence/cam1/run2/other.jpg", SECRET), "invalid")

    def test_extending_the_expiry_breaks_the_signature(self):
        exp, sig = cit.sign(self.PATH, SECRET, now=1000).split(".")
        self.assertEqual(cit.check_token(f"{int(exp) + 86400}.{sig}", self.PATH, SECRET, now=1000), "invalid")

    def test_a_changed_signature_is_invalid(self):
        exp, sig = cit.sign(self.PATH, SECRET).split(".")
        flipped = ("A" if sig[0] != "A" else "B") + sig[1:]
        self.assertEqual(cit.check_token(f"{exp}.{flipped}", self.PATH, SECRET), "invalid")

    def test_another_servers_secret_does_not_validate_it(self):
        tok = cit.sign(self.PATH, SECRET)
        self.assertEqual(cit.check_token(tok, self.PATH, b"a-different-secret-entirely"), "invalid")

    def test_malformed_tokens_are_invalid_and_never_raise(self):
        for bad in (None, "", ".", "abc", "1.", ".sig", "x.y", "99999999999999.sig", "১২৩.sig", "1.2.3", "-5.sig"):
            self.assertEqual(cit.check_token(bad, self.PATH, SECRET), "invalid", repr(bad))

    def test_a_forged_signature_is_not_told_it_has_expired(self):
        """Only a link that was really issued may learn that it expired; anything else learns nothing."""
        self.assertEqual(cit.check_token("1.AAAA", self.PATH, SECRET, now=10 ** 9), "invalid")

    def test_only_evidence_files_can_be_signed(self):
        for p in ("autozone/x.jpg", "surveillance.db", "/etc/passwd", "evidence/../surveillance.db",
                  "evidence/a/../../x", "evidence", "", None):
            self.assertIsNone(cit.sign(p, SECRET), repr(p))
            self.assertIsNone(cit.signed_url(p, SECRET), repr(p))

    def test_even_a_genuinely_signed_traversal_path_reaches_no_file(self):
        """Defence in depth: suppose the signing side were ever fooled. The serving side still refuses."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "evidence").mkdir()
            (root / "surveillance.db").write_bytes(b"every plate in the city")
            evil = "evidence/../surveillance.db"
            forged = f"2000000000.{cit._mac(SECRET, evil, 2000000000).decode()}"
            self.assertEqual(cit.check_token(forged, evil, SECRET, now=1), "ok")
            self.assertIsNone(cit.evidence_file(root, evil))

    def test_the_url_quotes_awkward_names_and_still_verifies(self):
        name = "evidence/cam 1/run#2/what?.jpg"
        url = cit.signed_url(name, SECRET)
        self.assertTrue(url.startswith("/m/"))
        self.assertNotIn(" ", url)
        self.assertNotIn("#", url)
        self.assertNotIn("?", url)
        _, _, token, rest = url.split("/", 3)
        self.assertEqual(unquote(rest), name)
        self.assertEqual(cit.check_token(token, unquote(rest), SECRET), "ok")


class SecretTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.dir = Path(self._dir.name)

    def test_the_environment_variable_wins(self):
        self.assertEqual(cit.media_secret(self.dir, {"MEDIA_SECRET": "x" * 24}), b"x" * 24)
        self.assertFalse((self.dir / cit.SECRET_FILE).exists(), "no file is written when the variable is set")

    def test_a_short_environment_secret_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "at least"):
            cit.media_secret(self.dir, {"MEDIA_SECRET": "short"})

    def test_a_key_is_generated_once_and_then_reused(self):
        first = cit.media_secret(self.dir, {}, log=lambda *_: None)
        self.assertGreaterEqual(len(first), 32)
        self.assertEqual(cit.media_secret(self.dir, {}, log=lambda *_: None), first)

    def test_each_installation_gets_a_different_key(self):
        with tempfile.TemporaryDirectory() as other:
            self.assertNotEqual(cit.media_secret(self.dir, {}, log=lambda *_: None),
                                cit.media_secret(other, {}, log=lambda *_: None))

    def _honours_modes(self):
        probe = self.dir / "probe"
        fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        ok = (os.stat(probe).st_mode & 0o777) == 0o600
        probe.unlink()
        return ok

    def test_the_key_file_is_created_owner_only(self):
        if not self._honours_modes():
            self.skipTest("this filesystem ignores Unix permissions")
        cit.media_secret(self.dir, {}, log=lambda *_: None)
        self.assertEqual(os.stat(self.dir / cit.SECRET_FILE).st_mode & 0o777, 0o600)

    def test_a_key_file_that_is_too_open_is_tightened(self):
        if not self._honours_modes():
            self.skipTest("this filesystem ignores Unix permissions")
        path = self.dir / cit.SECRET_FILE
        path.write_text("a" * 64 + "\n")
        os.chmod(path, 0o644)
        cit.media_secret(self.dir, {}, log=lambda *_: None)
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)

    def test_it_says_so_when_the_filesystem_will_not_make_the_file_private(self):
        path = self.dir / cit.SECRET_FILE
        path.write_text("a" * 64 + "\n")
        os.chmod(path, 0o644)
        said = []
        with mock.patch("surveillance.citation.os.chmod"):
            cit.media_secret(self.dir, {}, log=said.append)
        self.assertEqual(len(said), 1)
        self.assertIn("MEDIA_SECRET", said[0])

    def test_a_damaged_key_file_is_an_error_not_a_weak_key(self):
        (self.dir / cit.SECRET_FILE).write_text("abc")
        with mock.patch("surveillance.citation.time.sleep"), self.assertRaisesRegex(RuntimeError, "no usable key"):
            cit.media_secret(self.dir, {}, log=lambda *_: None)

    def test_processes_starting_together_end_up_with_one_key(self):
        barrier = threading.Barrier(8)

        def start(_):
            barrier.wait()
            return cit.media_secret(self.dir, {}, log=lambda *_: None)

        with ThreadPoolExecutor(8) as pool:
            keys = set(pool.map(start, range(8)))
        self.assertEqual(len(keys), 1)


class StoreScratch(Scratch):
    """A real store in a scratch folder, with cases whose evidence files exist on the disk."""

    def setUp(self):
        super().setUp()
        self.store = Store(self.root / "s.db")
        self.addCleanup(self.store.db.close)
        self.cam = self.store.add_camera("Farmgate 1", "file", "v.mp4", "Farmgate")
        uid = self.store.add_user("controller", "Sgt Rahman", username="rahman", password_hash="x")
        self.officer = {"user_id": uid, "name": "Sgt Rahman", "role": "controller", "address": "10.0.0.5"}
        aid = self.store.add_user("admin", "Admin", username="boss", password_hash="x")
        self.admin = {"user_id": aid, "name": "Admin", "role": "admin", "address": "10.0.0.1"}
        self.n = 0

    def case(self, code="V3", status="confirmed", plate="DHAKA METRO-GA 12-3456", plate_status="read"):
        """Insert a case with all of its evidence on the disk. Returns its id."""
        self.n += 1
        folder = self.root / "evidence" / f"cam{self.cam}" / "run1"
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{code}_t{self.n}_12.0s"
        rel = {}
        for key, name in (("snap", f"{stem}.jpg"), ("crop", f"{stem}_crop.jpg"), ("clip", f"{stem}.mp4"),
                          ("plate", f"plate_{self.n}.jpg"), ("stopped", f"{stem}_stopped.jpg")):
            (folder / name).write_bytes(f"{key}-{self.n}-".encode() * 400)
            rel[key] = str((folder / name).relative_to(self.root))
        rec = {"code": code, "violation": "Kerb parking", "track_id": self.n, "class": "Car",
               "class_confidence": 0.9, "video_time_s": 12.0, "started_s": 5.0, "wall_time": "2026-10-25 09:30:12",
               "zone": "z", "areas": "", "kph": 0.0, "details": {}}
        extra = [{"label": "stopped", "time_s": 5.0, "wall_time": "", "path": rel["stopped"]}]
        vid = self.store.add_violation(self.cam, None, None, rec, rel["snap"], rel["crop"], rel["clip"], "pending",
                                       extra)
        self.store.set_plate(vid, plate_status, "ঢাকা মেট্রো-গ ১২-৩৪৫৬", plate, 0.9,
                             rel["plate"] if plate_status in ("read", "partial") else None)
        self.store.review(vid, status, "", by=self.officer)
        return vid

    def issue(self, vid, **kw):
        return cit.issue(self.store, vid, self.officer, self.root, advisory=frozenset({"V2"}),
                         schedule=kw.pop("schedule", {}), **kw)


def fake_delivery(src, dst):
    Path(dst).write_bytes(Path(src).read_bytes()[:64])


class NumberingTests(StoreScratch):
    def test_numbers_have_the_documented_shape_and_run_in_sequence(self):
        got = [self.issue(self.case(), issued_at="2026-03-01T10:00:00")[0]["citation_no"] for _ in range(3)]
        self.assertEqual(got, ["DMP-2026-000001", "DMP-2026-000002", "DMP-2026-000003"])
        self.assertTrue(all(cit.NUMBER_RE.match(n) for n in got))

    def test_the_sequence_restarts_each_year_and_never_repeats(self):
        a = self.issue(self.case(), issued_at="2026-12-31T23:59:00")[0]["citation_no"]
        b = self.issue(self.case(), issued_at="2026-12-31T23:59:30")[0]["citation_no"]
        c = self.issue(self.case(), issued_at="2027-01-01T00:00:10")[0]["citation_no"]
        d = self.issue(self.case(), issued_at="2026-12-31T23:59:59")[0]["citation_no"]   # a late entry for 2026
        self.assertEqual([a, b, c, d], ["DMP-2026-000001", "DMP-2026-000002", "DMP-2027-000001", "DMP-2026-000003"])

    def test_the_prefix_is_configurable(self):
        row, _, _ = self.issue(self.case(), issued_at="2026-03-01T10:00:00", prefix="CMP")
        self.assertEqual(row["citation_no"], "CMP-2026-000001")

    def test_threads_never_get_the_same_number(self):
        vids = [self.case() for _ in range(12)]
        barrier = threading.Barrier(len(vids))

        def go(vid):
            barrier.wait()
            return self.store.create_citation(vid, self.officer, None, "h" * 64, "{}", None, "2026-03-01T10:00:00")[0]

        with ThreadPoolExecutor(len(vids)) as pool:
            rows = list(pool.map(go, vids))
        self.assertEqual(sorted(r["seq"] for r in rows), list(range(1, 13)))
        self.assertEqual(len({r["citation_no"] for r in rows}), 12)

    def test_separate_connections_never_get_the_same_number(self):
        """Two server processes are two connections to one file; SQLite's write lock is what keeps them apart."""
        other = Store(self.root / "s.db")
        self.addCleanup(other.db.close)
        vids = [self.case() for _ in range(16)]
        stores = [self.store, other]

        def go(i):
            return stores[i % 2].create_citation(vids[i], self.officer, None, "h" * 64, "{}", None,
                                                 "2026-03-01T10:00:00")[0]["seq"]

        with ThreadPoolExecutor(8) as pool:
            seqs = list(pool.map(go, range(len(vids))))
        self.assertEqual(sorted(seqs), list(range(1, 17)))

    def test_asking_for_one_case_many_times_at_once_makes_one_citation(self):
        vid = self.case()
        barrier = threading.Barrier(6)

        def go(_):
            barrier.wait()
            return self.store.create_citation(vid, self.officer, None, "h" * 64, "{}", None)

        with ThreadPoolExecutor(6) as pool:
            out = list(pool.map(go, range(6)))
        self.assertEqual(sum(1 for _, created in out if created), 1)
        self.assertEqual(len({row["citation_no"] for row, _ in out}), 1)

    def test_an_older_database_gains_the_table_when_it_is_opened(self):
        self.store.db.execute("DROP TABLE citations")
        self.store.db.commit()
        again = Store(self.root / "s.db")
        self.addCleanup(again.db.close)
        self.assertEqual(again.citations()[1], 0)


class IssuingTests(StoreScratch):
    def test_issuing_records_the_seal_the_issuer_and_an_unset_fine(self):
        vid = self.case()
        row, created, warnings = self.issue(vid)
        self.assertTrue(created)
        self.assertEqual(warnings, [])
        self.assertEqual((row["status"], row["violation_id"]), ("issued", vid))
        self.assertEqual((row["issued_by"], row["issued_by_name"]), (self.officer["user_id"], "Sgt Rahman"))
        self.assertIsNone(row["fine_bdt"])
        self.assertEqual(cit.view(row)["fine_text"], "set by the authority")
        self.assertEqual(row["bundle_sha256"], cit.sha256_text(row["manifest"]))
        self.assertEqual(cit.view(row)["manifest"]["violation_id"], vid)

    def test_issuing_twice_returns_the_same_citation_and_audits_once(self):
        vid = self.case()
        first = self.issue(vid)[0]
        row, created, _ = self.issue(vid)
        self.assertFalse(created)
        self.assertEqual(row["citation_no"], first["citation_no"])
        self.assertEqual(self.store.citations()[1], 1)
        self.assertEqual(len(self.store.audit_trail("citation", first["citation_no"])), 1)

    def test_a_speed_case_is_refused_and_nothing_is_stored(self):
        vid = self.case(code="V2")
        with self.assertRaisesRegex(cit.Refused, "advisory"):
            self.issue(vid)
        self.assertIsNone(self.store.citation_by_violation(vid))
        self.assertEqual(self.store.citations()[1], 0)

    def test_a_case_nobody_has_confirmed_is_refused(self):
        for status in ("pending", "rejected"):
            with self.assertRaisesRegex(cit.Refused, status):
                self.issue(self.case(status=status))

    def test_a_case_whose_plate_was_not_read_is_refused(self):
        for plate_status in ("unreadable", "pending", ""):
            with self.assertRaisesRegex(cit.Refused, "plate"):
                self.issue(self.case(plate_status=plate_status))

    def test_a_partly_read_plate_is_enough(self):
        self.assertTrue(self.issue(self.case(plate_status="partial"))[1])

    def test_a_clip_still_being_encoded_is_refused_until_it_is_finished(self):
        vid = self.case()
        final = self.root / self.store.violation(vid)["clip"]
        final.rename(final.with_suffix(".part.mp4"))
        with self.assertRaisesRegex(cit.Refused, "still being encoded"):
            self.issue(vid)
        final.with_suffix(".part.mp4").rename(final)
        self.assertTrue(self.issue(vid)[1])

    def test_a_clip_that_never_arrived_is_refused(self):
        vid = self.case()
        (self.root / self.store.violation(vid)["clip"]).unlink()
        with self.assertRaisesRegex(cit.Refused, "not on the disk"):
            self.issue(vid)

    def test_a_case_that_does_not_exist_is_a_404(self):
        with self.assertRaises(cit.Refused) as ctx:
            self.issue(999)
        self.assertEqual(ctx.exception.status, 404)

    def test_a_refused_attempt_is_audited_with_the_reason(self):
        """Somebody asking for a speed case to be cited is worth a row in the trail."""
        vid = self.case(code="V2")
        with self.assertRaises(cit.Refused):
            self.issue(vid)
        a = self.store.audit_trail("violation", vid)[0]
        self.assertEqual((a["action"], a["who"]), ("citation.refused", "Sgt Rahman"))
        self.assertIn("advisory", a["detail"])

    def test_the_fine_in_force_is_copied_onto_the_citation(self):
        vid = self.case()
        row = self.issue(vid, schedule={"V3": 1000})[0]
        self.assertEqual(row["fine_bdt"], 1000)
        self.assertEqual(cit.view(row)["fine_text"], "BDT 1,000")
        # the schedule changing later does not rewrite what was issued
        self.assertEqual(self.issue(vid, schedule={"V3": 5000})[0]["fine_bdt"], 1000)

    def test_issuing_is_audited_with_who_and_which_bundle(self):
        row = self.issue(self.case())[0]
        a = self.store.audit_trail("citation", row["citation_no"])[0]
        self.assertEqual((a["action"], a["who"], a["role"], a["address"]),
                         ("citation.issue", "Sgt Rahman", "controller", "10.0.0.5"))
        self.assertIn(row["bundle_sha256"], a["detail"])

    def test_a_withdrawn_citation_is_not_revived(self):
        vid = self.case()
        no = self.issue(vid)[0]["citation_no"]
        cit.withdraw(self.store, no, self.admin, "issued against the wrong vehicle")
        with self.assertRaisesRegex(cit.Refused, "withdrawn"):
            self.issue(vid)

    def test_a_shorter_copy_for_the_driver_is_made_beside_the_original_and_not_sealed(self):
        vid = self.case()
        row = self.issue(vid, delivery=fake_delivery)[0]
        clip = self.store.violation(vid)["clip"]
        self.assertEqual(row["delivery_clip"], clip[:-len(".mp4")] + "_delivery.mp4")
        self.assertTrue((self.root / row["delivery_clip"]).is_file())
        self.assertNotIn(row["delivery_clip"], row["manifest"])
        self.assertIn(clip, row["manifest"], "the original is what the seal covers")

    def test_a_failed_copy_does_not_stop_the_citation_and_is_retried_on_the_next_ask(self):
        def broken(src, dst):
            raise RuntimeError("ffmpeg exploded")

        vid = self.case()
        row, created, warnings = self.issue(vid, delivery=broken)
        self.assertTrue(created)
        self.assertIsNone(row["delivery_clip"])
        self.assertIn("ffmpeg exploded", warnings[0])
        again, created, warnings = self.issue(vid, delivery=fake_delivery)
        self.assertFalse(created)
        self.assertIsNotNone(again["delivery_clip"])
        self.assertEqual(again["citation_no"], row["citation_no"])
        self.assertEqual([a["action"] for a in self.store.audit_trail("citation", row["citation_no"])],
                         ["citation.delivery", "citation.issue"])


class WithdrawAndVerifyTests(StoreScratch):
    def issued(self):
        vid = self.case()
        return vid, self.issue(vid)[0]["citation_no"]

    def test_withdrawing_records_who_when_and_why(self):
        _, no = self.issued()
        row, changed = cit.withdraw(self.store, no, self.admin, "plate misread on review")
        self.assertTrue(changed)
        self.assertEqual((row["status"], row["withdrawn_by_name"], row["withdraw_reason"]),
                         ("withdrawn", "Admin", "plate misread on review"))
        self.assertTrue(row["withdrawn_at"])
        a = self.store.audit_trail("citation", no)[0]
        self.assertEqual((a["action"], a["who"], a["detail"]), ("citation.withdraw", "Admin", "plate misread on review"))

    def test_a_reason_is_required(self):
        _, no = self.issued()
        for blank in ("", "   ", "no", None):
            with self.assertRaises(cit.Refused) as ctx:
                cit.withdraw(self.store, no, self.admin, blank)
            self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(self.store.citation(no)["status"], "issued")

    def test_withdrawing_twice_changes_nothing_and_is_audited_once(self):
        _, no = self.issued()
        cit.withdraw(self.store, no, self.admin, "plate misread on review")
        _, changed = cit.withdraw(self.store, no, self.admin, "plate misread on review")
        self.assertFalse(changed)
        self.assertEqual([a["action"] for a in self.store.audit_trail("citation", no)],
                         ["citation.withdraw", "citation.issue"])

    def test_a_paid_citation_cannot_be_withdrawn_here(self):
        _, no = self.issued()
        self.store.x("UPDATE citations SET status='paid' WHERE citation_no=?", (no,))
        with self.assertRaisesRegex(cit.Refused, "paid"):
            cit.withdraw(self.store, no, self.admin, "changed our minds")

    def test_an_unknown_number_is_a_404(self):
        for call in (lambda: cit.withdraw(self.store, "DMP-2026-999999", self.admin, "whatever reason"),
                     lambda: cit.verify_citation(self.store, "DMP-2026-999999", self.officer, self.root)):
            with self.assertRaises(cit.Refused) as ctx:
                call()
            self.assertEqual(ctx.exception.status, 404)

    def test_verifying_an_untouched_citation_passes_and_is_audited(self):
        _, no = self.issued()
        r = cit.verify_citation(self.store, no, self.officer, self.root)
        self.assertTrue(r["ok"])
        self.assertEqual(r["counts"]["match"], 5)
        a = self.store.audit_trail("citation", no)[0]
        self.assertEqual((a["action"], a["who"]), ("citation.verify", "Sgt Rahman"))
        self.assertTrue(a["detail"].startswith("intact"))

    def test_verifying_after_a_file_changed_names_the_file_and_says_so_in_the_audit(self):
        vid, no = self.issued()
        crop = self.root / self.store.violation(vid)["crop"]
        crop.write_bytes(crop.read_bytes() + b"!")
        r = cit.verify_citation(self.store, no, self.officer, self.root)
        self.assertFalse(r["ok"])
        self.assertEqual([f["path"] for f in r["files"] if f["status"] == "mismatch"], [self.store.violation(vid)["crop"]])
        self.assertTrue(self.store.audit_trail("citation", no)[0]["detail"].startswith("NOT intact"))

    def test_a_manifest_edited_in_the_database_is_caught(self):
        _, no = self.issued()
        text = self.store.citation(no)["manifest"]
        self.store.x("UPDATE citations SET manifest=? WHERE citation_no=?", (text.replace("V3", "V1", 1), no))
        self.assertEqual(cit.verify_citation(self.store, no, self.officer, self.root)["bundle"]["status"], "mismatch")

    def test_the_seal_survives_the_case_being_deleted(self):
        """A citation outlives the operational data: its record stays and verification says what is gone."""
        vid, no = self.issued()
        self.store.x("DELETE FROM violations WHERE id=?", (vid,))
        self.assertEqual(self.store.citation(no)["violation_id"], vid)
        r = cit.verify_citation(self.store, no, self.officer, self.root)
        self.assertFalse(r["ok"])
        self.assertEqual(set(r["case"].values()), {"missing"})

    def test_a_changed_plate_on_the_case_is_reported_against_the_sealed_one(self):
        vid, no = self.issued()
        self.store.set_plate(vid, "read", "x", "DHAKA METRO-KA 99-9999", 0.9, None)
        self.assertEqual(cit.verify_citation(self.store, no, self.officer, self.root)["case"]["plate"], "mismatch")

    def test_verification_ignores_the_drivers_shorter_copy(self):
        vid = self.case()
        row = self.issue(vid, delivery=fake_delivery)[0]
        (self.root / row["delivery_clip"]).unlink()
        self.assertTrue(cit.verify_citation(self.store, row["citation_no"], self.officer, self.root)["ok"])


def ffmpeg_or_skip():
    from surveillance import clips
    if clips.imageio_ffmpeg is None:
        raise unittest.SkipTest("imageio-ffmpeg is not installed")
    try:
        clips.imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:      # pragma: no cover - depends on the machine
        raise unittest.SkipTest(f"no ffmpeg binary: {e}")
    return clips


def make_source(path, w, h, frames, fps=30):
    """A clip written the way the recorder writes them: JPEG frames through clips.write_mp4. The frames carry
    texture and a moving block, so the encoder has real work to do and a size comparison means something."""
    import cv2
    import numpy as np
    from surveillance import clips
    rng = np.random.default_rng(1)
    base = np.linspace(0, 255, w, dtype=np.float32)[None, :, None] * np.ones((h, 1, 3), np.float32)
    noise = rng.normal(0, 12, (h, w, 3)).astype(np.float32)    # drawn once, then shifted per frame: much cheaper
    jpegs = []
    for i in range(frames):
        img = base + np.roll(noise, 5 * i, axis=1)
        x = int(i * (w - 200) / frames)
        img[h // 3:h // 3 + 150, x:x + 200] = (30, 60, 200)
        ok, buf = cv2.imencode(".jpg", np.clip(img, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpegs.append(buf.tobytes())
    clips.write_mp4(Path(path), jpegs, fps)


def video_facts(path):
    import cv2
    c = cv2.VideoCapture(str(path))
    facts = {"w": int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), "h": int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)),
             "fps": c.get(cv2.CAP_PROP_FPS), "frames": int(c.get(cv2.CAP_PROP_FRAME_COUNT)), "decoded": 0}
    while True:
        ok, frame = c.read()
        if not ok:
            break
        facts["decoded"] += frame is not None
    c.release()
    return facts


class DeliveryClipTests(unittest.TestCase):
    """The smaller copy for the driver, made with the ffmpeg that ships in the virtualenv."""

    @classmethod
    def setUpClass(cls):
        cls.clips = ffmpeg_or_skip()
        cls._dir = tempfile.TemporaryDirectory()
        cls.dir = Path(cls._dir.name)
        cls.big, cls.small = cls.dir / "big.mp4", cls.dir / "small.mp4"
        make_source(cls.big, 1920, 1080, 30)
        make_source(cls.small, 640, 360, 30)
        cls.out = cls.dir / "big_delivery.mp4"
        cls.before = cit.sha256_file(cls.big)
        cls.clips.make_delivery_clip(cls.big, cls.out)

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def test_the_copy_is_smaller_than_the_original(self):
        self.assertLess(self.out.stat().st_size, self.big.stat().st_size)

    def test_it_is_no_taller_than_720_and_keeps_the_picture_shape(self):
        f = video_facts(self.out)
        self.assertEqual((f["w"], f["h"]), (1280, 720))

    def test_it_keeps_every_frame_and_the_frame_rate_and_plays_to_the_end(self):
        a, b = video_facts(self.big), video_facts(self.out)
        self.assertEqual(b["frames"], a["frames"])
        self.assertAlmostEqual(b["fps"], a["fps"], delta=0.5)
        self.assertEqual(b["decoded"], a["frames"], "every frame decodes")

    def test_it_is_h264_with_the_index_at_the_front(self):
        import subprocess
        info = subprocess.run([self.clips.imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(self.out)],
                              capture_output=True, text=True).stderr
        self.assertIn("Video: h264", info)
        data = self.out.read_bytes()
        self.assertGreater(data.find(b"moov"), 0)
        self.assertLess(data.find(b"moov"), data.find(b"mdat"), "moov first: playback starts before the download ends")

    def test_the_original_is_left_exactly_as_it_was(self):
        self.assertEqual(cit.sha256_file(self.big), self.before)

    def test_a_clip_that_is_already_small_is_not_enlarged(self):
        dst = self.dir / "small_delivery.mp4"
        self.clips.make_delivery_clip(self.small, dst)
        f = video_facts(dst)
        self.assertEqual((f["w"], f["h"]), (640, 360))

    def test_nothing_half_written_is_left_when_it_fails(self):
        bad = self.dir / "not_a_video.mp4"
        bad.write_bytes(b"this is not a video")
        dst = self.dir / "bad_delivery.mp4"
        with self.assertRaisesRegex(RuntimeError, "ffmpeg failed"):
            self.clips.make_delivery_clip(bad, dst)
        self.assertFalse(dst.exists())
        self.assertFalse(dst.with_suffix(".part.mp4").exists())

    def test_it_will_not_overwrite_the_original(self):
        with self.assertRaises(ValueError):
            self.clips.make_delivery_clip(self.small, self.small)

    def test_a_missing_ffmpeg_is_an_error_the_caller_can_handle(self):
        with mock.patch.object(self.clips, "imageio_ffmpeg", None), self.assertRaisesRegex(RuntimeError, "ffmpeg"):
            self.clips.make_delivery_clip(self.small, self.dir / "x_delivery.mp4")


class DeliveryIssuingTests(StoreScratch):
    def test_issuing_with_the_real_transcode_seals_the_original_and_makes_a_playable_copy(self):
        clips = ffmpeg_or_skip()
        vid = self.case()
        original = self.root / self.store.violation(vid)["clip"]
        make_source(original, 640, 360, 30)
        row, _, warnings = self.issue(vid, delivery=clips.make_delivery_clip)
        self.assertEqual(warnings, [])
        copy = self.root / row["delivery_clip"]
        self.assertEqual(video_facts(copy)["decoded"], 30)
        self.assertIn(cit.sha256_file(original), row["manifest"], "the seal is over the original, not the copy")
        self.assertNotIn(cit.sha256_file(copy), row["manifest"])
        self.assertTrue(cit.verify_citation(self.store, row["citation_no"], self.officer, self.root)["ok"])


class PortalAndListingTests(StoreScratch):
    def keys(self):
        return ["DHAKAMETROGA123456"]

    def test_a_driver_never_gets_a_speed_case_even_when_it_is_confirmed(self):
        speed, kerb = self.case(code="V2"), self.case(code="V3")
        ids = [r["id"] for r in self.store.violations_for_plates(self.keys())[0]]
        self.assertEqual(ids, [kerb])
        self.assertNotIn(speed, ids)
        self.assertEqual(self.store.violations_for_plates(self.keys())[1], 1)

    def test_the_exclusion_follows_the_configured_policy_unless_overridden_on_purpose(self):
        self.case(code="V2")
        self.assertEqual(self.store.violations_for_plates(self.keys(), exclude_codes=())[1], 1)
        self.assertEqual(self.store.violations_for_plates(self.keys(), exclude_codes=("V2",))[1], 0)

    def test_a_case_row_carries_its_citation(self):
        cited, plain = self.case(), self.case()
        no = self.issue(cited, schedule={"V3": 700}, delivery=fake_delivery)[0]["citation_no"]
        rows = {r["id"]: r for r in self.store.violations_for_plates(self.keys())[0]}
        self.assertEqual((rows[cited]["citation_no"], rows[cited]["citation_status"], rows[cited]["citation_fine_bdt"]),
                         (no, "issued", 700))
        self.assertTrue(rows[cited]["delivery_clip"].endswith("_delivery.mp4"))
        self.assertIsNone(rows[plain]["citation_no"])
        staff = {r["id"]: r for r in self.store.violations(limit=10)}
        self.assertEqual(staff[cited]["citation_no"], no)
        self.assertIsNone(staff[plain]["citation_no"])
        self.assertEqual(self.store.violation(cited)["citation_status"], "issued")

    def test_the_listing_filters_by_status_and_plate_and_leaves_out_the_manifest(self):
        a, b = self.case(), self.case(plate="DHAKA METRO-KA 77-7777")
        na, nb = self.issue(a)[0]["citation_no"], self.issue(b)[0]["citation_no"]
        cit.withdraw(self.store, nb, self.admin, "plate misread on review")
        rows, total = self.store.citations()
        self.assertEqual((total, [r["citation_no"] for r in rows]), (2, [nb, na]))
        self.assertNotIn("manifest", rows[0])
        self.assertEqual([r["citation_no"] for r in self.store.citations(status="issued")[0]], [na])
        self.assertEqual([r["citation_no"] for r in self.store.citations(plate="ka-77 7777")[0]], [nb])
        self.assertEqual(rows[0]["code"], "V3")
        self.assertEqual(rows[0]["camera_name"], "Farmgate 1")

    def test_a_standing_citation_is_reported_against_its_camera_until_withdrawn(self):
        vid = self.case()
        no = self.issue(vid)[0]["citation_no"]
        self.assertEqual(self.store.live_citations_for_camera(self.cam), [no])
        cit.withdraw(self.store, no, self.admin, "plate misread on review")
        self.assertEqual(self.store.live_citations_for_camera(self.cam), [])


# ------------------------------------------------------------------ the HTTP surface

def asgi_call(app, method, path, headers=None, body=b""):
    """Call the app in-process, because httpx (and so FastAPI's TestClient) is not installed.
    `path` is the decoded path, as a server hands it to the app. Returns (status, headers, body)."""
    import asyncio
    path, _, query = path.partition("?")
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
             "method": method, "scheme": "http", "path": path, "raw_path": path.encode("utf-8"),
             "query_string": query.encode(), "root_path": "", "client": ("203.0.113.9", 50000),
             "server": ("testserver", 80),
             "headers": [(b"host", b"testserver")] + [(k.lower().encode(), v.encode())
                                                      for k, v in (headers or {}).items()]}
    sent = []

    async def run():
        pending = [{"type": "http.request", "body": body, "more_body": False}]
        finished = asyncio.Event()

        async def receive():
            if pending:
                return pending.pop(0)
            await finished.wait()       # a real server holds the connection until the response is complete
            return {"type": "http.disconnect"}

        async def send(msg):
            sent.append(msg)
            if msg["type"] == "http.response.body" and not msg.get("more_body"):
                finished.set()

        await app(scope, receive, send)

    asyncio.run(run())
    start = next(m for m in sent if m["type"] == "http.response.start")
    return (start["status"], {k.decode().lower(): v.decode() for k, v in start["headers"]},
            b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body"))


class ServerScratch(StoreScratch):
    """The real FastAPI app over a scratch store and data folder, with a session for each kind of person."""

    @classmethod
    def setUpClass(cls):
        import surveillance.server as srv
        cls.srv = srv

    def setUp(self):
        super().setUp()
        srv = self.srv
        patches = [mock.patch.object(srv, "store", self.store), mock.patch.object(srv, "DATA", self.root),
                   mock.patch.object(srv, "make_delivery_clip", fake_delivery),
                   mock.patch.object(srv.sessions, "store", self.store),
                   mock.patch.dict(srv._media_key, {"key": SECRET}, clear=True)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.driver = self.store.add_user("driver", "Karim", licence_no="DK1234567", phone="+8801712345678")
        self.store.add_vehicle("DHAKAMETROGA123456", self.driver, plate_en="DHAKA METRO-GA 12-3456", verified=1)
        self.ctrl = self.sign_in(self.officer["user_id"], "controller")
        self.boss = self.sign_in(self.admin["user_id"], "admin")
        self.drv = self.sign_in(self.driver, "driver")

    def sign_in(self, user_id, role):
        token = auth.new_token()
        self.store.add_session(auth.token_fingerprint(token), user_id, role, time.time() + 600)
        return {"cookie": f"tsid={token}"}

    def call(self, method, path, who=None, body=None, extra=None, follow=False):
        """`follow`: do what a browser does with a redirect, up to twice (FastAPI redirects /x to /x/ when only
        the second matches a route)."""
        headers, data = dict(who or {}, **(extra or {})), b""
        if body is not None:
            data = json.dumps(body).encode()
            headers["content-type"] = "application/json"
        out = asgi_call(self.srv.app, method, path, headers, data)
        for _ in range(2 if follow else 0):
            if out[0] not in (301, 302, 307, 308):
                break
            out = asgi_call(self.srv.app, method, urlsplit(out[1]["location"]).path, headers, data)
        return out

    def get_json(self, path, who):
        status, _, body = self.call("GET", path, who)
        return status, json.loads(body)

    def post_json(self, path, who, body=None):
        status, _, out = self.call("POST", path, who, body)
        return status, json.loads(out)


class StaffMediaTests(ServerScratch):
    """The folders the dashboard reads from used to be public mounts. They now need a staff session."""

    def setUp(self):
        super().setUp()
        self.vid = self.case()
        self.rel = self.store.violation(self.vid)["snapshot"]
        self.path = "/media/" + self.rel
        self.bytes = (self.root / self.rel).read_bytes()
        (self.root / "autozone").mkdir(exist_ok=True)
        (self.root / "autozone" / "p.jpg").write_bytes(b"learned zones")
        (self.root / "surveillance.db").write_bytes(b"every plate in the city")

    def test_evidence_needs_a_session(self):
        self.assertEqual(self.call("GET", self.path)[0], 401)

    def test_a_driver_is_refused_even_with_a_session(self):
        self.assertEqual(self.call("GET", self.path, self.drv)[0], 403)

    def test_a_controller_and_an_admin_get_the_file(self):
        for who in (self.ctrl, self.boss):
            status, _, body = self.call("GET", self.path, who)
            self.assertEqual((status, body), (200, self.bytes))

    def test_a_response_is_private_and_not_sniffable(self):
        h = self.call("GET", self.path, self.ctrl)[1]
        self.assertIn("private", h["cache-control"])
        self.assertEqual(h["x-content-type-options"], "nosniff")

    def test_learned_zone_pictures_are_guarded_the_same_way(self):
        self.assertEqual(self.call("GET", "/media/autozone/p.jpg")[0], 401)
        self.assertEqual(self.call("GET", "/media/autozone/p.jpg", self.drv)[0], 403)
        self.assertEqual(self.call("GET", "/media/autozone/p.jpg", self.ctrl)[2], b"learned zones")

    def test_an_anonymous_caller_learns_nothing_about_which_folders_exist(self):
        self.assertEqual(self.call("GET", "/media/videos/x.mp4")[0], 401)
        self.assertEqual(self.call("GET", "/media/evidence/nothing-here.jpg")[0], 401)

    def test_the_database_and_other_folders_are_not_reachable_even_when_signed_in(self):
        for path in ("/media/surveillance.db", "/media/videos/x.mp4", "/media/evidence", "/media/evidence/",
                     "/media/evidence/../surveillance.db", "/media/autozone/../surveillance.db",
                     "/media/evidence/cam1/run1/../../../surveillance.db", "/media/evidence//etc/passwd"):
            status, _, body = self.call("GET", path, self.boss, follow=True)
            self.assertEqual(status, 404, path)
            self.assertNotIn(b"every plate", body, path)

    def test_a_symlink_inside_the_folder_cannot_lead_out_of_it(self):
        try:
            (self.root / "evidence" / "cam1" / "db.jpg").symlink_to(self.root / "surveillance.db")
        except OSError:
            self.skipTest("symlinks are not available here")
        status, _, body = self.call("GET", "/media/evidence/cam1/db.jpg", self.boss)
        self.assertEqual(status, 404)
        self.assertNotIn(b"every plate", body)

    def test_a_video_can_be_fetched_in_ranges_so_it_can_be_scrubbed(self):
        clip = self.store.violation(self.vid)["clip"]
        size = (self.root / clip).stat().st_size
        status, h, body = self.call("GET", "/media/" + clip, self.ctrl, extra={"range": "bytes=0-9"})
        self.assertEqual((status, len(body), h["content-range"]), (206, 10, f"bytes 0-9/{size}"))


class SignedLinkRouteTests(ServerScratch):
    def setUp(self):
        super().setUp()
        self.vid = self.case()
        self.rel = self.store.violation(self.vid)["snapshot"]
        self.bytes = (self.root / self.rel).read_bytes()
        (self.root / "surveillance.db").write_bytes(b"every plate in the city")
        self.url = unquote(self.srv.shared_link(self.rel))

    def forged(self, path, ttl=600):
        exp = int(time.time()) + ttl
        return f"/m/{exp}.{cit._mac(SECRET, path, exp).decode()}/{path}"

    def test_a_link_opens_its_file_with_no_session_at_all(self):
        status, _, body = self.call("GET", self.url)
        self.assertEqual((status, body), (200, self.bytes))

    def test_an_expired_link_is_gone(self):
        old = unquote(cit.signed_url(self.rel, SECRET, ttl=60, now=time.time() - 3600))
        status, _, body = self.call("GET", old)
        self.assertEqual(status, 410)
        self.assertNotEqual(body, self.bytes)

    def test_a_link_pointed_at_another_file_is_refused(self):
        other = self.url.rsplit("/", 1)[0] + "/" + Path(self.store.violation(self.vid)["crop"]).name
        self.assertEqual(self.call("GET", other)[0], 403)

    def test_a_link_with_a_changed_signature_or_expiry_is_refused(self):
        token = self.url.split("/")[2]
        exp, sig = token.split(".")
        flipped = ("A" if sig[0] != "A" else "B") + sig[1:]
        for bad in (f"{exp}.{flipped}", f"{int(exp) + 86400}.{sig}", "garbage", "0.0", f"{exp}."):
            status = self.call("GET", self.url.replace(token, bad))[0]
            self.assertEqual(status, 403, bad)

    def test_an_unsigned_guess_and_a_signed_miss_are_told_apart_only_by_whoever_holds_a_link(self):
        missing = "evidence/cam1/run1/nope.jpg"
        self.assertEqual(self.call("GET", "/m/1.AAAA/" + missing)[0], 403)
        self.assertEqual(self.call("GET", unquote(cit.signed_url(missing, SECRET)))[0], 404)

    def test_a_genuinely_signed_path_that_climbs_out_of_the_folder_still_serves_nothing(self):
        for path in ("evidence/../surveillance.db", "evidence/cam1/run1/../../../surveillance.db"):
            status, _, body = self.call("GET", self.forged(path))
            self.assertEqual(status, 404, path)
            self.assertNotIn(b"every plate", body)

    def test_a_signed_path_outside_evidence_serves_nothing(self):
        (self.root / "autozone").mkdir(exist_ok=True)
        (self.root / "autozone" / "p.jpg").write_bytes(b"learned zones")
        for path in ("autozone/p.jpg", "surveillance.db"):
            self.assertEqual(self.call("GET", self.forged(path))[0], 404, path)

    def test_a_response_carries_no_referrer_and_a_short_cache_life(self):
        h = self.call("GET", self.url)[1]
        self.assertEqual(h["referrer-policy"], "no-referrer")
        self.assertIn("private", h["cache-control"])
        self.assertLessEqual(int(h["cache-control"].split("max-age=")[1]), 300)

    def test_a_video_link_can_be_scrubbed(self):
        clip = self.store.violation(self.vid)["clip"]
        url = unquote(self.srv.shared_link(clip))
        status, h, body = self.call("GET", url, extra={"range": "bytes=5-14"})
        self.assertEqual((status, len(body)), (206, 10))

    def test_the_signing_key_is_made_on_first_use_and_then_reused(self):
        with mock.patch.dict(self.srv._media_key, {}, clear=True), mock.patch.dict(os.environ):
            os.environ.pop("MEDIA_SECRET", None)
            a, b = self.srv.media_key(), self.srv.media_key()
        self.assertEqual(a, b)
        self.assertTrue((self.root / cit.SECRET_FILE).exists())


class CitationApiTests(ServerScratch):
    def setUp(self):
        super().setUp()
        self.vid = self.case()
        self.url = f"/api/violations/{self.vid}/citation"

    def issue_over_http(self, vid=None):
        return self.post_json(f"/api/violations/{vid or self.vid}/citation", self.ctrl)

    def test_a_controller_issues_a_citation(self):
        status, out = self.issue_over_http()
        self.assertEqual(status, 200)
        self.assertTrue(cit.NUMBER_RE.match(out["citation_no"]))
        self.assertEqual((out["status"], out["created"], out["fine_text"]), ("issued", True, "set by the authority"))
        self.assertEqual(out["manifest"]["violation_id"], self.vid)
        self.assertEqual(out["bundle_sha256"], cit.sha256_text(self.store.citation(out["citation_no"])["manifest"]))
        self.assertTrue(out["delivery_clip"].endswith("_delivery.mp4"))

    def test_asking_again_returns_the_same_citation(self):
        first = self.issue_over_http()[1]
        status, again = self.issue_over_http()
        self.assertEqual((status, again["created"], again["citation_no"]), (200, False, first["citation_no"]))
        self.assertEqual(self.store.citations()[1], 1)

    def test_nobody_without_a_staff_session_can_use_any_citation_endpoint(self):
        no = self.issue_over_http()[1]["citation_no"]
        calls = [("POST", self.url), ("GET", "/api/citations"), ("GET", f"/api/citations/{no}"),
                 ("GET", f"/api/citations/{no}/verify"), ("POST", f"/api/citations/{no}/withdraw")]
        for method, path in calls:
            self.assertEqual(self.call(method, path)[0], 401, path)
            self.assertEqual(self.call(method, path, self.drv)[0], 403, path)

    def test_a_speed_case_cannot_be_cited_and_the_reason_is_given(self):
        status, out = self.issue_over_http(self.case(code="V2"))
        self.assertEqual(status, 409)
        self.assertIn("advisory", out["detail"])
        self.assertEqual(self.store.citations()[1], 0)

    def test_a_case_nobody_has_confirmed_cannot_be_cited(self):
        status, out = self.issue_over_http(self.case(status="pending"))
        self.assertEqual(status, 409)
        self.assertIn("pending", out["detail"])

    def test_a_case_that_does_not_exist_is_404(self):
        self.assertEqual(self.issue_over_http(999)[0], 404)

    def test_only_an_administrator_can_withdraw(self):
        no = self.issue_over_http()[1]["citation_no"]
        path = f"/api/citations/{no}/withdraw"
        self.assertEqual(self.post_json(path, self.ctrl, {"reason": "plate misread on review"})[0], 403)
        self.assertEqual(self.post_json(path, self.boss, {})[0], 400)
        status, out = self.post_json(path, self.boss, {"reason": "plate misread on review"})
        self.assertEqual((status, out["status"], out["changed"]), (200, "withdrawn", True))
        self.assertEqual(self.post_json(path, self.boss, {"reason": "plate misread on review"})[1]["changed"], False)

    def test_the_list_filters_by_status_and_never_carries_the_manifest(self):
        a, b = self.issue_over_http()[1]["citation_no"], self.issue_over_http(self.case())[1]["citation_no"]
        self.post_json(f"/api/citations/{b}/withdraw", self.boss, {"reason": "issued against the wrong car"})
        status, out = self.get_json("/api/citations", self.ctrl)
        self.assertEqual((status, out["total"], [i["citation_no"] for i in out["items"]]), (200, 2, [b, a]))
        self.assertNotIn("manifest", out["items"][0])
        self.assertEqual(out["items"][0]["fine_text"], "set by the authority")
        self.assertEqual([i["citation_no"] for i in self.get_json("/api/citations?status=issued", self.ctrl)[1]["items"]], [a])
        self.assertEqual(self.get_json("/api/citations?status=bogus", self.ctrl)[0], 400)

    def test_the_detail_carries_the_manifest_and_the_case(self):
        no = self.issue_over_http()[1]["citation_no"]
        status, out = self.get_json(f"/api/citations/{no}", self.ctrl)
        self.assertEqual(status, 200)
        self.assertEqual(len(out["manifest"]["files"]), 5)
        self.assertEqual((out["case"]["id"], out["case"]["code"]), (self.vid, "V3"))

    def test_verification_over_http_names_the_file_that_changed(self):
        no = self.issue_over_http()[1]["citation_no"]
        self.assertTrue(self.get_json(f"/api/citations/{no}/verify", self.ctrl)[1]["ok"])
        crop = self.store.violation(self.vid)["crop"]
        (self.root / crop).write_bytes(b"swapped")
        out = self.get_json(f"/api/citations/{no}/verify", self.ctrl)[1]
        self.assertFalse(out["ok"])
        self.assertEqual([f["path"] for f in out["files"] if f["status"] == "mismatch"], [crop])

    def test_malformed_and_unknown_numbers_are_404(self):
        for no in ("nonsense", "DMP-2026-999999", "dmp-2026-000001", "DMP-26-1"):
            self.assertEqual(self.call("GET", f"/api/citations/{no}", self.ctrl)[0], 404, no)
            self.assertEqual(self.call("GET", f"/api/citations/{no}/verify", self.ctrl)[0], 404, no)

    def test_a_decision_cannot_change_underneath_a_standing_citation(self):
        no = self.issue_over_http()[1]["citation_no"]
        patch = lambda who: self.call("PATCH", f"/api/violations/{self.vid}", who, {"status": "rejected", "note": "x"})
        status, _, body = patch(self.ctrl)
        self.assertEqual(status, 409)
        self.assertIn(no, json.loads(body)["detail"])
        self.assertEqual(self.store.violation(self.vid)["status"], "confirmed")
        self.post_json(f"/api/citations/{no}/withdraw", self.boss, {"reason": "plate misread on review"})
        self.assertEqual(patch(self.ctrl)[0], 200)

    def test_a_camera_with_standing_citations_cannot_be_deleted_or_emptied(self):
        no = self.issue_over_http()[1]["citation_no"]
        folder = self.root / "evidence" / f"cam{self.cam}"
        self.assertEqual(self.call("DELETE", f"/api/cameras/{self.cam}", self.boss)[0], 409)
        self.assertEqual(self.call("DELETE", f"/api/cameras/{self.cam}/violations", self.boss)[0], 409)
        self.assertTrue(folder.exists(), "the evidence is still on the disk")
        self.assertIsNotNone(self.store.violation(self.vid))
        self.post_json(f"/api/citations/{no}/withdraw", self.boss, {"reason": "plate misread on review"})
        self.assertEqual(self.call("DELETE", f"/api/cameras/{self.cam}/violations", self.boss)[0], 200)

    def test_every_citation_action_leaves_a_row_in_the_audit_trail(self):
        no = self.issue_over_http()[1]["citation_no"]
        self.get_json(f"/api/citations/{no}/verify", self.ctrl)
        self.post_json(f"/api/citations/{no}/withdraw", self.boss, {"reason": "plate misread on review"})
        self.assertEqual([(a["action"], a["who"]) for a in self.store.audit_trail("citation", no)],
                         [("citation.withdraw", "Admin"), ("citation.verify", "Sgt Rahman"),
                          ("citation.issue", "Sgt Rahman")])

    def test_the_dashboard_is_told_which_codes_are_advisory(self):
        with mock.patch.object(self.srv.vlm_mod, "backend_status", return_value={"enabled": False}):
            status, out = self.get_json("/api/meta", self.ctrl)
        self.assertEqual((status, out["advisory_codes"]), (200, sorted(cit.ADVISORY_CODES)))

    def test_the_case_rows_the_dashboard_reads_carry_the_citation(self):
        no = self.issue_over_http()[1]["citation_no"]
        out = self.get_json(f"/api/violations/{self.vid}", self.ctrl)[1]
        self.assertEqual((out["citation_no"], out["citation_status"]), (no, "issued"))
        items = self.get_json("/api/violations?status=confirmed", self.ctrl)[1]["items"]
        self.assertEqual([i["citation_no"] for i in items], [no])


class NoticeTests(ServerScratch):
    def setUp(self):
        super().setUp()
        self.vid = self.case()

    def page(self, vid=None, who=None):
        status, _, body = self.call("GET", f"/notice/{vid or self.vid}", who or self.ctrl)
        return status, body.decode("utf-8")

    def test_a_speed_case_is_never_printed_as_a_notice(self):
        status, body = self.page(self.case(code="V2"))
        self.assertEqual(status, 409)
        self.assertIn("advisory", body)

    def test_a_cited_case_shows_the_number_the_seal_and_how_to_check_it(self):
        row = self.issue(self.vid, schedule={})[0]
        status, html = self.page()
        self.assertEqual(status, 200)
        for want in (row["citation_no"], row["bundle_sha256"], "set by the authority", "Sgt Rahman",
                     f"/api/citations/{row['citation_no']}/verify", 'class="citation issued"'):
            self.assertIn(want, html)
        for f in json.loads(row["manifest"])["files"]:
            self.assertIn(f["sha256"], html, "every sealed file's hash is on the page")
        self.assertNotIn("{{", html)

    def test_the_fine_shows_when_the_authority_has_set_one(self):
        self.issue(self.vid, schedule={"V3": 1500})
        self.assertIn("BDT 1,500", self.page()[1])

    def test_an_uncited_case_says_no_citation_exists(self):
        html = self.page()[1]
        self.assertIn('class="citation none"', html)
        self.assertIn("No citation has been issued", html)
        self.assertNotIn("set by the authority", html)

    def test_a_withdrawn_citation_is_marked_void_with_its_reason(self):
        no = self.issue(self.vid)[0]["citation_no"]
        cit.withdraw(self.store, no, self.admin, "plate misread on review")
        html = self.page()[1]
        for want in ("WITHDRAWN", "void", "plate misread on review", 'class="citation withdrawn"'):
            self.assertIn(want, html)

    def test_text_is_escaped_and_never_expanded_as_a_template_field(self):
        self.store.review(self.vid, "confirmed", "<b>x</b> {{plate}} {{sealed}}", by=self.officer)
        html = self.page()[1]
        self.assertIn("&lt;b&gt;x&lt;/b&gt; {{plate}} {{sealed}}", html)
        self.assertNotIn("<b>x</b>", html)

    def test_the_notice_is_for_staff_only(self):
        self.assertEqual(self.call("GET", f"/notice/{self.vid}")[0], 401)
        self.assertEqual(self.call("GET", f"/notice/{self.vid}", self.drv)[0], 403)


class PortalTests(ServerScratch):
    def setUp(self):
        super().setUp()
        self.vid = self.case()
        self.row = self.issue(self.vid, schedule={}, delivery=fake_delivery)[0]

    def listing(self, who=None):
        status, out = self.get_json("/api/portal/violations", who or self.drv)
        self.assertEqual(status, 200)
        return out

    def links(self, item):
        urls = [item["snapshot"], item["crop"], item["plate_img"], item["clip"]]
        return urls + [e["url"] for e in item["evidence"]]

    def test_files_are_signed_links_and_never_paths(self):
        status, _, raw = self.call("GET", "/api/portal/violations", self.drv)
        item = json.loads(raw)["items"][0]
        for url in self.links(item):
            self.assertTrue(url.startswith("/m/"), url)
        self.assertEqual(len(item["evidence"]), 1)
        self.assertNotIn('"evidence/', raw.decode(), "no bare path reaches the driver")

    def test_every_link_in_the_response_opens_its_file_without_a_session(self):
        item = self.listing()["items"][0]
        for url in self.links(item):
            self.assertEqual(self.call("GET", unquote(url))[0], 200, url)

    def test_the_clip_is_the_shorter_copy_not_the_original(self):
        clip = unquote(self.listing()["items"][0]["clip"])
        self.assertTrue(clip.endswith("_delivery.mp4"), clip)

    def test_a_case_that_has_not_been_cited_gets_the_original_clip_and_no_number(self):
        self.store.x("DELETE FROM citations")
        item = self.listing()["items"][0]
        self.assertIsNone(item["citation"])
        self.assertFalse(unquote(item["clip"]).endswith("_delivery.mp4"))

    def test_the_citation_is_shown_with_its_fine_in_words(self):
        c = self.listing()["items"][0]["citation"]
        self.assertEqual((c["number"], c["status"], c["fine_text"]), (self.row["citation_no"], "issued", "set by the authority"))

    def test_a_withdrawn_citation_is_shown_as_withdrawn(self):
        cit.withdraw(self.store, self.row["citation_no"], self.admin, "plate misread on review")
        self.assertEqual(self.listing()["items"][0]["citation"]["status"], "withdrawn")

    def test_speed_cases_do_not_appear_and_cannot_be_opened_by_their_number(self):
        speed = self.case(code="V2")
        self.assertEqual([i["id"] for i in self.listing()["items"]], [self.vid])
        self.assertEqual(self.call("GET", f"/api/portal/violation/{speed}", self.drv)[0], 404)
        self.assertEqual(self.call("GET", f"/api/portal/violation/{self.vid}", self.drv)[0], 200)

    def test_a_file_that_is_gone_is_left_out_rather_than_linked(self):
        (self.root / self.store.violation(self.vid)["crop"]).unlink()
        item = self.listing()["items"][0]
        self.assertIsNone(item["crop"])
        self.assertIsNotNone(item["snapshot"])

    def test_another_driver_sees_nothing_of_it(self):
        other = self.store.add_user("driver", "Other", licence_no="DK7654321", phone="+8801700000000")
        who = self.sign_in(other, "driver")
        self.assertEqual(self.listing(who)["items"], [])
        self.assertEqual(self.call("GET", f"/api/portal/violation/{self.vid}", who)[0], 404)

    def test_what_the_reviewers_wrote_stays_internal(self):
        item = self.listing()["items"][0]
        for hidden in ("note", "reviewed_by", "reviewed_by_name", "camera_id", "track_id", "citation_no", "manifest",
                       "bundle_sha256", "delivery_clip"):
            self.assertNotIn(hidden, item)

    def test_a_driver_cannot_reach_the_staff_routes_with_the_same_session(self):
        for path in ("/api/violations", "/api/citations", f"/notice/{self.vid}",
                     "/media/" + self.store.violation(self.vid)["snapshot"]):
            self.assertEqual(self.call("GET", path, self.drv)[0], 403, path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
