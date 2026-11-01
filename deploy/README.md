# Putting the dashboard on a test server

One small Linux VM runs the dashboard over HTTPS so anyone with the link can log in and upload recorded videos.
It is a test box: no camera feeds, no real citizen data, and payments use the built-in test gateway.

```
laptop ──push.sh (rsync)──▶ VM: ~/dhaka-traffic        code + detector/plate/helmet/ReID weights (~270 MB)
                               ~/dhaka-traffic-data   database, uploads, evidence (never touched by a push)
browser ──HTTPS──▶ Caddy :443 ──▶ dashboard 127.0.0.1:8000 (systemd service "dhaka-traffic")
```

Files here:

| File | Runs on | Does |
|---|---|---|
| `push.sh` | laptop | copies the working tree (uncommitted changes included) to the VM; restarts the service once it is installed |
| `setup.sh` | VM, once | packages, swap, venv (CPU-only PyTorch), secrets file, systemd service, Caddy + HTTPS |

## 1. Get the credit (no credit card needed)

**Azure for Students:** go to <https://azure.microsoft.com/free/students>, sign in and verify with your NSU email
(`@northsouth.edu`). You get **US$100 for 12 months** and can renew it each year while you are a student. It asks
for no card, and when the credit runs out the subscription is suspended; it does not bill you.

(DigitalOcean's US$200 from the GitHub Student Pack also works with these scripts, but it needs a card or PayPal
on file. See [DigitalOcean instead](#digitalocean-instead) if your dual-currency debit card is accepted.)

## 2. Create the VM (Azure portal)

**Virtual machines → Create → Azure virtual machine:**

| Field | Value |
|---|---|
| Region | Southeast Asia (Singapore) or Central India. Student subscriptions only allow some regions; pick any one that is allowed |
| Image | **Ubuntu Server 24.04 LTS – x64 Gen2** |
| Size | **B2s** (2 vCPU, 4 GiB), enough without the VLM. **B2ms** (8 GiB) if you have spare credit |
| Authentication | SSH public key, username `azureuser`. "Use existing public key" → paste `~/.ssh/id_ed25519.pub` (make one with `ssh-keygen -t ed25519` if you have none) |
| Inbound ports | **SSH (22), HTTP (80), HTTPS (443)** |
| Disks | OS disk **64 GiB, Standard SSD**. The default 30 GiB fills up after a few uploads |

After it is created, open the VM's **Public IP address** resource → **Configuration** → **DNS name label**, e.g.
`dhaka-traffic-g05`. The server's name becomes `dhaka-traffic-g05.<region>.cloudapp.azure.com`. Caddy gets the
HTTPS certificate for that name.

**Stop the VM paying for itself while nobody is testing:**

- VM → **Auto-shutdown**: on, e.g. 02:00, time zone Dhaka.
- Use the portal's **Stop** button (it *deallocates*; compute billing stops). `sudo shutdown` inside the VM does
  not stop the bill.
- **Cost Management → Budgets**: alert at US$50 and US$80.

Rough cost, B2s (check the Azure pricing calculator for your region): about US$30–40/month if it runs 24/7,
including the disk and IP. The disk and IP (about US$8/month) keep billing while it is stopped. Running only when
you test (around 8 h a day) makes the credit last the semester.

## 3. Deploy

From the project folder on your laptop (use the VM's public IP):

```bash
./deploy/push.sh azureuser@<public-ip>
ssh azureuser@<public-ip> bash dhaka-traffic/deploy/setup.sh dhaka-traffic-g05.<region>.cloudapp.azure.com
```

`setup.sh` takes 5–10 minutes the first time (PyTorch download). At the end it prints the URL and the **admin
password, once**. That password is also in `/etc/dhaka-traffic.env` on the VM.

Open the URL, log in as `admin`, create the police/driver test accounts, then upload a video from the dashboard.

## 4. Updating

After changing code on the laptop:

```bash
./deploy/push.sh azureuser@<public-ip>
```

It copies the changed files, installs any new Python packages and restarts the dashboard (about a minute until it
answers again). Uploaded videos and cases stay.

## 5. Running it

On the VM (`ssh azureuser@<public-ip>`):

| Task | Command |
|---|---|
| Logs (live) | `journalctl -u dhaka-traffic -f` |
| Driver login codes | `journalctl -u dhaka-traffic \| grep "login code"`: there is no SMS gateway, so codes go to the log |
| Restart | `sudo systemctl restart dhaka-traffic` |
| Settings / secrets | `sudoedit /etc/dhaka-traffic.env`, then restart |
| Disk space | `df -h /` and the dashboard's Storage page |
| HTTPS problems | `journalctl -u caddy -n 30`: usually port 80/443 closed or the DNS name not set yet |
| Back up the data to the laptop | (on the laptop) `rsync -az azureuser@<public-ip>:dhaka-traffic-data/ ./backup-data/` |

## What is different from the laptop

- **No VLM by default** (`VLM_BACKEND=off`): the Qwen models need more RAM than a 4 GB VM has. To get the Explain
  button, set `VLM_BACKEND=cloud` and an `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY` + `OPENAI_BASE_URL`) in
  `/etc/dhaka-traffic.env`.
- **Slower analysis.** Two vCPUs give roughly a third of the laptop's throughput (`docs/CAPACITY.md`). Upload short
  clips, and set the camera's "Frames analysed per second" to 15 for test runs.
- **Fresh database.** The laptop's `data/` (test accounts, demo cases) is not copied. Only `admin` exists at first.
- **No CoreML packages.** `push.sh` leaves out `*.mlpackage`, so the helmet rule uses `helmet_cls.pt` (CoreML only
  predicts on macOS).
- **Secure settings.** `OTP_DELIVERY=log`, Secure cookies, `TRUST_PROXY_HEADER=1` behind Caddy. Never copy
  `OTP_DELIVERY=echo` or `INSECURE_COOKIES=1` from `tools/run_dashboard.sh` to the server: it is on the public
  internet.

## DigitalOcean instead

Same scripts, three differences:

1. **Create → Droplet:** Ubuntu 24.04 x64, Basic / Regular, **4 GB / 2 vCPU (US$24/month)**, region Bangalore or
   Singapore, your SSH key. Then **Networking → Firewalls**: allow 22, 80, 443.
2. **A droplet logs you in as root.** Make a normal user first:
   `ssh root@<ip>`, then `adduser deploy && usermod -aG sudo deploy && rsync -a ~/.ssh /home/deploy/ && chown -R deploy: /home/deploy/.ssh`.
   Use `deploy@<ip>` everywhere above.
3. **No free DNS name.** Use `<ip-with-dashes>.sslip.io` (e.g. `165-22-10-5.sslip.io`): it resolves to that IP and
   Caddy can get a certificate for it.

DigitalOcean bills a powered-off droplet in full. To stop paying, snapshot it and destroy it.
