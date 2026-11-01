#!/usr/bin/env bash
# One-time setup of a test server (Ubuntu 24.04, x86-64, 4 GB RAM or more). Run ON THE SERVER, as the normal login
# user (not root), after deploy/push.sh has copied the code to ~/dhaka-traffic:
#
#   bash ~/dhaka-traffic/deploy/setup.sh traffic-nsu.southeastasia.cloudapp.azure.com
#
# The argument is the public host name Caddy gets an HTTPS certificate for (see deploy/README.md). Re-running is safe:
# packages and the venv are updated, the secrets file is kept, the service is restarted.
#
# What it sets up:
#   - 4 GB swap file           pip and the first model load briefly need more than a 4 GB VM has
#   - .venv                    CPU-only PyTorch first, so pip does not pull several GB of CUDA libraries
#   - /etc/dhaka-traffic.env   MEDIA_SECRET and the first admin password, generated once, root-only (0600)
#   - dhaka-traffic.service    the dashboard on 127.0.0.1:8000, restarted on failure and at boot
#   - Caddy                    HTTPS on 443 (Let's Encrypt), proxying to the dashboard
#
# Data (database, uploaded videos, evidence) lives in ~/dhaka-traffic-data, outside the code folder, so a push can
# never overwrite or delete it.
set -euo pipefail

if [[ $EUID -eq 0 ]]; then
    echo "Run this as your normal login user (it uses sudo where needed), not as root." >&2
    echo "On a DigitalOcean droplet, create one first:  adduser deploy && usermod -aG sudo deploy" >&2
    exit 1
fi

APP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$HOME/dhaka-traffic-data"
ENV_FILE=/etc/dhaka-traffic.env
UNIT=/etc/systemd/system/dhaka-traffic.service
DOMAIN="${1:-}"
if [[ -z "$DOMAIN" && -f /etc/caddy/Caddyfile ]]; then
    DOMAIN="$(awk 'NR==1 && $2=="{" {print $1}' /etc/caddy/Caddyfile)"
fi
if [[ -z "$DOMAIN" ]]; then
    echo "usage: $0 <public host name>    e.g. traffic-nsu.southeastasia.cloudapp.azure.com" >&2
    exit 2
fi

step() { echo; echo "==> $*"; }

step "System packages"
sudo apt-get update -q
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3-venv python3-pip libgl1 libglib2.0-0 caddy curl openssl

step "Swap"
if ! swapon --show=NAME --noheadings | grep -q .; then
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile >/dev/null
    sudo swapon /swapfile
    grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
    echo "4 GB swap file added"
else
    echo "swap already present, left alone"
fi

step "Python environment (a few minutes the first time)"
cd "$APP"
[[ -x .venv/bin/python ]] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -q -r requirements.txt
chmod +x start.sh tools/*.sh deploy/*.sh 2>/dev/null || true

step "Data folder"
mkdir -p "$DATA"
chmod 700 "$DATA"

step "Secrets"
ADMIN_PASSWORD_NEW=""
if sudo test -f "$ENV_FILE"; then
    echo "$ENV_FILE exists, kept (edit it with: sudoedit $ENV_FILE)"
else
    ADMIN_PASSWORD_NEW="$(openssl rand -base64 18 | tr -d '/+=' | cut -c1-16)"
    sudo install -m 600 -o root -g root /dev/null "$ENV_FILE"
    sudo tee "$ENV_FILE" >/dev/null <<EOF
# Dashboard settings for this server. Root-only; systemd reads it before starting the service.
# After editing:  sudo systemctl restart dhaka-traffic

# Signs evidence links. Changing it invalidates every link already handed out.
MEDIA_SECRET=$(openssl rand -hex 32)

# Used only when the database has no accounts yet (first start). Change it from the dashboard afterwards.
ADMIN_USER=admin
ADMIN_PASSWORD=$ADMIN_PASSWORD_NEW

# Data outside the code folder, so deploy/push.sh never touches it.
TRAFFIC_DATA_DIR=$DATA

# Links in emails and payment return URLs.
PUBLIC_BASE_URL=https://$DOMAIN

# Caddy sits in front and sets X-Forwarded-For, so the rate limiter sees real client addresses.
TRUST_PROXY_HEADER=1

# Driver login codes go to the server log (read them with: journalctl -u dhaka-traffic | grep "login code").
# Never set OTP_DELIVERY=echo or INSECURE_COOKIES=1 here: this server is on the public internet.
OTP_DELIVERY=log

# The local VLM needs more RAM than a small VM has. off = no Explain button; cloud = set a key below.
VLM_BACKEND=off
# VLM_BACKEND=cloud
# ANTHROPIC_API_KEY=
# or an OpenAI-compatible endpoint:
# OPENAI_API_KEY=
# OPENAI_BASE_URL=
# OPENAI_MODEL=

# Mail (optional; without SMTP_HOST emails are written to the data folder's outbox/).
# SMTP_HOST=
# SMTP_PORT=587
# SMTP_USER=
# SMTP_PASSWORD=
# SMTP_FROM=
# SMTP_STARTTLS=1
EOF
    echo "created $ENV_FILE"
fi

step "Service"
sudo tee "$UNIT" >/dev/null <<EOF
[Unit]
Description=Dhaka Traffic Surveillance dashboard
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$APP
EnvironmentFile=$ENV_FILE
ExecStart=$APP/.venv/bin/python app.py --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=5
# Analysis can take every core; keep SSH and Caddy responsive.
Nice=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable -q dhaka-traffic
sudo systemctl restart dhaka-traffic

step "HTTPS (Caddy)"
sudo tee /etc/caddy/Caddyfile >/dev/null <<EOF
$DOMAIN {
	encode gzip
	reverse_proxy 127.0.0.1:8000 {
		# live view is an endless MJPEG stream and Explain progress is SSE: pass bytes through as they come
		flush_interval -1
	}
}
EOF
sudo systemctl enable -q caddy
sudo systemctl reload caddy 2>/dev/null || sudo systemctl restart caddy

step "Waiting for the dashboard to start (loading PyTorch takes a while)"
for _ in $(seq 1 90); do
    code="$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/login || true)"
    [[ "$code" == "200" ]] && break
    sleep 2
done
if [[ "$code" != "200" ]]; then
    echo "The dashboard did not answer on :8000. Look at:  journalctl -u dhaka-traffic -n 50" >&2
    exit 1
fi

echo
echo "  Done.  https://$DOMAIN"
if [[ -n "$ADMIN_PASSWORD_NEW" ]]; then
    echo "  First login:  admin / $ADMIN_PASSWORD_NEW     (shown once; also in $ENV_FILE)"
fi
echo "  If the HTTPS page does not load yet: ports 80 and 443 must be open in the cloud firewall, and the name must"
echo "  point at this machine. Caddy's log:  journalctl -u caddy -n 30"
