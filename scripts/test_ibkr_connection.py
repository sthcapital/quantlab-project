from argparse import ArgumentParser

from ib_insync import IB

from quantlab.providers.ibkr import ping_tws
from quantlab.utils import get_config


def main() -> None:
    ibkr_cfg = get_config("ibkr")
    parser = ArgumentParser(description="Test TWS / IB Gateway connectivity.")
    parser.add_argument("--host", default=ibkr_cfg["host"])
    parser.add_argument("--port", type=int, default=ibkr_cfg["port"])
    parser.add_argument("--client-id", type=int, default=ibkr_cfg["client_id"])
    args = parser.parse_args()

    print(f"Pinging {args.host}:{args.port} ...", end=" ", flush=True)
    if not ping_tws(args.host, args.port):
        print("UNREACHABLE")
        raise SystemExit(
            f"\nTWS / IB Gateway is not reachable at {args.host}:{args.port}.\n"
            "Check that TWS is running, API is enabled, and this IP is in Trusted IPs."
        )
    print("OK")

    ib = IB()
    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=10)
        print(f"CONNECTED")
        print(f"  isConnected   = {ib.isConnected()}")
        print(f"  serverVersion = {ib.client.serverVersion()}")
    finally:
        if ib.isConnected():
            ib.disconnect()
            print("DISCONNECTED")


if __name__ == "__main__":
    main()
