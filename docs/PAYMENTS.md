# Fines, owner registry, online payment, invoices and email (WP18)

Code: `surveillance/{payments,api_payments,registry,invoice,mailer}.py`; pages `web/payments.*` (admin) and
`web/portal.*` (driver). Tests: `tests/test_{payments,registry,invoice,mailer}.py`.

## What it does

* **Fines** (admin, Fines tab): whole-taka amount and legal reference per violation code. By default no code has an
  amount (`FINE_SCHEDULE_BDT` is the fallback, the database wins). An amount with no legal reference is shown as
  "placeholder - check against the Road Transport Act 2018". Advisory codes (V2, V10, V11) cannot have a fine. A
  citation with no amount cannot be paid ("amount not set by the authority"). When someone first tries to pay a
  citation issued before its amount existed, the amount in force is copied onto it (audited).
* **Registry** (admin): plate to owner name, mobile, email, licence number. CSV import is previewed (dry run) then
  applied. The registry email receives the notices. A driver whose licence number is on a registry row may pay and
  download invoices for that plate, in addition to their verified claimed vehicles (`registry.driver_plate_keys`).
* **Payment state** of a citation: `unpaid`, `paid`, `paused` (a dispute is open), `cancelled` (withdrawn, which an
  accepted dispute does). A dispute that was upheld makes it payable again, due at least 7 days after the decision.
  Due date = issue date + `PAYMENT_DUE_DAYS` (30).
* **Test gateway**: two buttons, no card or wallet fields, labelled "Test - no money moves". On when
  `PAYMENTS_TEST_GATEWAY=1`, or when SSLCommerz is not configured; **always off when `SSLCOMMERZ_LIVE=1`** (even if
  forced), so a pretend payment can never be taken for a real one.
* **SSLCommerz**: session created server-side, the driver is redirected to `GatewayPageURL`. A payment counts only
  when the validation API (asked with `val_id`) says `VALID`/`VALIDATED` and its `tran_id`, `amount` and currency
  (`BDT`) equal what we stored; a gateway "risky" flag holds it. Return URLs and the IPN only trigger that check
  (nothing in their query/body is believed except the `val_id` used as the question). Settling is idempotent; each
  attempt has its own transaction id. If the validation API is unreachable the IPN answers 503 so the gateway retries.
* **Invoice**: small hand-written PDF 1.4 (Helvetica), downloadable by the owner and by staff once paid.
* **Email**: `data/outbox/*.eml` and the Outbox tab when `SMTP_HOST` is unset; otherwise queued and sent by a
  background thread, retried 1, 2, 4... minutes (cap 1 h, 8 tries), then "failed" with a Retry button.

## Environment

| Variable | Meaning |
|---|---|
| `SSLCOMMERZ_STORE_ID`, `SSLCOMMERZ_STORE_PASSWORD` | gateway credentials (environment only; never sent to the browser) |
| `SSLCOMMERZ_LIVE=1` | real money (securepay host); default is the sandbox |
| `PUBLIC_BASE_URL` | the public https origin; the gateway's return/IPN URLs and email links are built from it (never from the request) |
| `PAYMENTS_TEST_GATEWAY=1` | force the test gateway on (ignored when live) |
| `PAYMENT_DUE_DAYS` | days to pay after issue (default 30) |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM`, `SMTP_STARTTLS` | mail; port 465 uses SSL; STARTTLS default on |

## Wiring in `surveillance/server.py`

```python
from . import api_payments                                   # with the other imports
# after the other include_router calls:
pay = api_payments.build(store, DATA, current_user, require, CONTROLLER, ADMIN, DRIVER)
app.include_router(pay.router)
cit.on_issued.append(pay.on_issued)                          # citation-issued email
disputes_mod.on_decided.append(pay.on_decided)               # dispute-decided email
# lifespan: pay.mailer.start() before `yield`, pay.mailer.stop() after it
PUBLIC_API |= {"/api/payments/sslcommerz/success", "/api/payments/sslcommerz/fail",
               "/api/payments/sslcommerz/cancel", "/api/payments/sslcommerz/ipn"}
```

No new `DRIVER_API` entry is needed: every driver endpoint is under `/api/portal/citations`, already listed. Nav link
(admin only): `<a href="/payments" data-route="payments">Fines &amp; payments</a>`.

The portal's own citation list and dispute filing still use `server._my_plate_keys` (claimed and verified plates); if
the registry licence match should apply there too, make that function return `registry.driver_plate_keys(store, user)`.

## What a real deployment still needs

HTTPS and a real `PUBLIC_BASE_URL`; the authority's actual amounts and legal references; SSLCommerz merchant
credentials and a registered IPN URL; refunds and chargebacks (a payment that settles after a citation was withdrawn is
held as "rejected" for manual refund; paying then opening a dispute is not reconciled); a real SMTP account; a decision
on how long payment records and the registry (personal data) are kept.
