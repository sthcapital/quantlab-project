from ib_insync import IB


def main() -> None:
    ib = IB()

    try:
        ib.connect("127.0.0.1", 7497, clientId=1, timeout=10)
        print("CONNECTED")
        print(f"isConnected={ib.isConnected()}")
        print(f"serverVersion={ib.client.serverVersion()}")
    finally:
        if ib.isConnected():
            ib.disconnect()
            print("DISCONNECTED")


if __name__ == "__main__":
    main()
