#!/usr/bin/env bash
# Start the dashboard for local development.
#
#   ./tools/run_dashboard.sh            # port 8000
#   ./tools/run_dashboard.sh --port 8010
#
# The evidence-link signing key (MEDIA_SECRET) is kept in ~/.config/dhaka-traffic/media_secret, not in data/:
# on this machine data/ sits on an NTFS (fuseblk) mount that ignores chmod, so a key file there is readable by
# every local user. The key is created once (32 random bytes, mode 0600) and reused, so signed links survive a
# restart. Delete the file to rotate the key; every outstanding link then stops working.
#
# OTP_DELIVERY=echo prints driver login codes to the console and INSECURE_COOKIES=1 allows plain-HTTP cookies.
# Both are for local development only — never set them on a hosted instance.
set -euo pipefail
cd "$(dirname "$0")/.."

secret_dir="${XDG_CONFIG_HOME:-$HOME/.config}/dhaka-traffic"
secret_file="$secret_dir/media_secret"
if [[ -z "${MEDIA_SECRET:-}" ]]; then
  if [[ ! -s "$secret_file" ]]; then
    mkdir -p "$secret_dir" && chmod 700 "$secret_dir"
    (umask 077 && head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$secret_file")
  fi
  MEDIA_SECRET="$(cat "$secret_file")"
fi
export MEDIA_SECRET
export OTP_DELIVERY="${OTP_DELIVERY:-echo}"
export INSECURE_COOKIES="${INSECURE_COOKIES:-1}"
# Links in emails and payment return URLs are built from this, never from the request's Host header. Locally the
# emails land in data/outbox/ (no SMTP_HOST) and payments use the built-in test gateway (no SSLCommerz keys).
export PUBLIC_BASE_URL="${PUBLIC_BASE_URL:-http://localhost:8000}"

exec ./.venv/bin/python app.py "$@"
