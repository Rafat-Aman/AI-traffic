"""A payment invoice as a small, valid PDF, written by hand.

Why hand-written: the invoice is a handful of lines of text, and a PDF library would be a new dependency to
install, audit and keep patched on a machine that otherwise needs none. PDF 1.4 with the standard Helvetica
font (one of the 14 fonts every reader has built in, so nothing is embedded) is a few dozen lines of code.

What it is not: it is not a tax invoice and carries no signature. It is the receipt for a fine the driver paid,
with the facts a dispute about the payment would start from. Text is Latin-1 only (the standard fonts have no
Bangla); anything outside that is replaced with "?" rather than producing a file readers refuse to open.
"""

LINE_HEIGHT = 18
PAGE_W, PAGE_H = 595, 842          # A4 in points


def _escape(text):
    """A string as a PDF literal: Latin-1 only, with the three characters that have meaning escaped."""
    t = "".join(ch if ch == "\t" or 32 <= ord(ch) < 256 and ord(ch) != 127 else "?" for ch in str(text))
    return t.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(title, lines, footer=""):
    """Bytes of a one-page PDF: a title, then `lines` (strings, or (label, value) pairs), then a small footer."""
    ops = ["BT", "/F2 18 Tf", f"50 {PAGE_H - 70} Td", f"({_escape(title)}) Tj", "ET"]
    y = PAGE_H - 110
    for line in lines:
        text = f"{line[0]}: {line[1]}" if isinstance(line, (tuple, list)) else str(line)
        ops += ["BT", "/F1 12 Tf", f"50 {y} Td", f"({_escape(text[:110])}) Tj", "ET"]
        y -= LINE_HEIGHT
    if footer:
        ops += ["BT", "/F1 9 Tf", "50 50 Td", f"({_escape(footer[:130])}) Tj", "ET"]
    stream = "\n".join(ops).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] /Contents 6 0 R "
         "/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> >>").encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("ascii") + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode("ascii")
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").encode("ascii")
    return bytes(out)


def invoice_pdf(p, c, plate, violation_title):
    """The invoice for payment row `p` of citation row `c`."""
    test = p["provider"] == "test"
    lines = [("Case number", c["citation_no"]), ("Number plate", plate or "-"),
             ("Violation", violation_title or "-"), ("Amount", f"BDT {p['amount_bdt']:,}"),
             ("Payment provider", "Test gateway" if test else "SSLCommerz"),
             ("Transaction id", p["tran_id"]), ("Paid at", (p["paid_at"] or "").replace("T", " "))]
    if p.get("bank_tran_id"):
        lines.append(("Bank transaction id", p["bank_tran_id"]))
    if test:
        lines += ["", "Test - no money moved"]
    return make_pdf("Payment invoice - Dhaka Traffic Surveillance", lines,
                    "Receipt for a traffic fine. Quote the case number when you contact the traffic authority.")
