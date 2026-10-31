import argparse
import json
import threading
import webbrowser
from pathlib import Path

from .config import Settings


def main():
    parser = argparse.ArgumentParser(description="DemandCast forecasting and monitoring")
    parser.add_argument("command", choices=["serve", "train", "verify"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--open", action="store_true", help="Open the localhost dashboard in your browser")
    parser.add_argument("--no-tracking", action="store_true", help="Skip optional local MLflow experiment tracking")
    args = parser.parse_args()
    if args.command == "train":
        from .training import train
        report = train(args.root, track=not args.no_tracking)
        print(json.dumps({"model": report["selection"]["selected_candidate"],
                          "test": report["selected_test_metrics"], "registry": report["registry"]}, indent=2))
    elif args.command == "verify":
        from .service import ForecastService
        service = ForecastService(Settings.from_env())
        print(json.dumps({"bundle_verified": True, **service.config()}, indent=2))
    else:
        import uvicorn
        if args.open:
            threading.Timer(1.5, webbrowser.open, args=(f"http://127.0.0.1:{args.port}",)).start()
        uvicorn.run("demandcast.api:create_app", factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
