"""
Dhaka Traffic Surveillance - dashboard server

    python app.py                 # then open http://localhost:8000
    python app.py --port 8080

The dashboard is also reachable from a phone or another computer on the same Wi-Fi at the address printed below.
"""

import argparse
import logging

import uvicorn


class _QuietStreams(logging.Filter):
    """Hide access-log lines for the endless live-view / status polling requests."""

    def filter(self, record):
        msg = record.getMessage()
        return not any(p in msg for p in ("/stream", "/api/status", "/api/events"))


def main():
    p = argparse.ArgumentParser(description="Traffic surveillance dashboard")
    p.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = reachable from other devices on your network")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    from surveillance.server import app, lan_url  # imports torch/ultralytics: after argument parsing

    logging.getLogger("uvicorn.access").addFilter(_QuietStreams())
    print("\n  Dhaka Traffic Surveillance")
    print(f"  On this Mac:        http://localhost:{args.port}")
    if args.host == "0.0.0.0" and (url := lan_url(args.port)):
        print(f"  From phone / LAN:   {url}")
    print()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
