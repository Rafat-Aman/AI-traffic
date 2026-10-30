"""The hand-written invoice PDF: starts like a PDF, and its cross-reference table points at real objects."""

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surveillance import invoice  # noqa: E402

PAID = {"provider": "test", "amount_bdt": 1500, "tran_id": "TVABCDEF0123456789", "paid_at": "2026-10-30T10:00:00",
        "bank_tran_id": ""}
CIT = {"citation_no": "DMP-2026-000001"}


def check_structure(test, pdf):
    """Parse the file the way a reader does: trailer -> startxref -> xref table -> every object at its offset."""
    test.assertTrue(pdf.startswith(b"%PDF-1.4"))
    test.assertTrue(pdf.rstrip().endswith(b"%%EOF"))
    start = int(re.search(rb"startxref\n(\d+)\n%%EOF", pdf).group(1))
    test.assertTrue(pdf[start:].startswith(b"xref\n"))
    n = int(re.match(rb"xref\n0 (\d+)\n", pdf[start:]).group(1))
    entries = re.findall(rb"(\d{10}) (\d{5}) ([nf]) \n", pdf[start:])
    test.assertEqual(len(entries), n)
    for num, (off, _gen, kind) in enumerate(entries):
        if kind == b"n":
            test.assertTrue(pdf[int(off):].startswith(f"{num} 0 obj\n".encode()), num)
    test.assertIn(b"/Root 1 0 R", pdf)
    length = int(re.search(rb"/Length (\d+)", pdf).group(1))
    body = pdf.split(b"stream\n", 1)[1]
    test.assertEqual(body[length:length + 10], b"\nendstream")


class InvoiceTests(unittest.TestCase):
    def test_structure(self):
        pdf = invoice.invoice_pdf(PAID, CIT, "DHAKA METRO-GA 12-3456", "Kerb parking")
        check_structure(self, pdf)

    def test_content_has_the_facts_and_the_test_label(self):
        pdf = invoice.invoice_pdf(PAID, CIT, "DHAKA METRO-GA 12-3456", "Kerb parking")
        for text in (b"DMP-2026-000001", b"DHAKA METRO-GA 12-3456", b"Kerb parking", b"BDT 1,500",
                     b"Test gateway", b"TVABCDEF0123456789", b"2026-10-30 10:00:00", b"Test - no money moved"):
            self.assertIn(text, pdf)

    def test_a_real_payment_does_not_carry_the_test_label(self):
        pdf = invoice.invoice_pdf({**PAID, "provider": "sslcommerz"}, CIT, "X-1234", "Kerb parking")
        self.assertNotIn(b"no money moved", pdf)
        self.assertIn(b"SSLCommerz", pdf)

    def test_special_characters_cannot_break_the_file(self):
        pdf = invoice.make_pdf("t) (", ["a\\b (c) বাংলা", ("k", "v")])
        check_structure(self, pdf)
        self.assertIn(b"a\\\\b \\(c\\) ?????", pdf)


if __name__ == "__main__":
    unittest.main()
