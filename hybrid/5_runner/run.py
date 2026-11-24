"""
Main orchestrator for Personal Health Assistant.

Architecture:
Internet
   |
Cloudflare Tunnel
   |
Gateway (127.0.0.1:7860)
   |
   |--- llama-server (127.0.0.1:8080)
   |    SmolLM2-360M + SFT-QLoRA
   |
   |--- llama-server (127.0.0.1:8081)
        SmolLM2-1.7B-Instruct

First run:
python hybrid/5_runner/run.py --admin your@email.com

Next runs:
python hybrid/5_runner/run.py

Local testing:
python hybrid/5_runner/run.py --no-tunnel
"""

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path


# --------------------------------------------------
# PATHS
# --------------------------------------------------

HERE = Path(__file__).resolve().parent
HYBRID = HERE.parent

BIN = HYBRID / "bin"
DATA = HYBRID / "data"
SETTINGS = DATA / "settings.env"
MODELS = HYBRID / "models"

# Model 1: SmolLM2-360M + SFT-QLoRA
MODEL_1 = MODELS / "medicalai-360m-sft-qlora-q8_0.gguf"

# Model 2: SmolLM2-1.7B-Instruct
MODEL_2 = MODELS / "smollm2-1.7b-instruct-q8_0.gguf"

LLAMA_SERVER = BIN / "llama" / "llama-server.exe"
CLOUDFLARED = BIN / "cloudflared.exe"


# --------------------------------------------------
# DEFAULT SETTINGS
# --------------------------------------------------

DEFAULTS = {
    "ADMIN_EMAILS": "",
    "SIGNUP_CODE": "",
    "RATE_LIMIT": "20",
    "DAILY_LIMIT": "500",
    "PARALLEL": "2",
    "THREADS": str(max(1, (os.cpu_count() or 4) - 2)),
    "UPDATE_VERCEL": "1"
}

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

def log(*args):
    line = time.strftime("%Y-%m-%d %H:%M:%S ")
    line += " ".join(map(str, args))

    print(line, flush=True)

    with open(DATA / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

def load_settings(admin):
    settings = dict(DEFAULTS)

    if SETTINGS.exists():
        for line in SETTINGS.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                settings[key.strip()] = value.strip()

    if admin:
        settings["ADMIN_EMAILS"] = admin

    SETTINGS.write_text(
        "# Personal Health Assistant server settings\n"
        "# Restart run.py after changing these values.\n"
        + "".join(f"{key}={value}\n" for key, value in settings.items()),
        encoding="utf-8"
    )

    return settings


# --------------------------------------------------
# NETWORK
# --------------------------------------------------

def port_busy(port):
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_http(url, seconds):
    for _ in range(seconds):
        try:
            with urllib.request.urlopen(url, timeout=3):
                return True
        except (OSError, TimeoutError):
            time.sleep(1)

    return False


# --------------------------------------------------
# PROCESS MANAGER
# --------------------------------------------------

class Proc:
    """Starts, monitors and stops an individual process."""

    def __init__(self, name, cmd, cwd=None, env=None, on_line=None):
        self.name = name
        self.cmd = cmd
        self.cwd = cwd
        self.env = env
        self.on_line = on_line
        self.p = None

    def start(self):
        log_file = DATA / f"{self.name}.log"

        try:
            process = subprocess.Popen(
                self.cmd,
                cwd=self.cwd,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=NO_WINDOW
            )
        except Exception as e:
            log(self.name, "failed to start:", repr(e))
            raise

        self.p = process

        def pump():
            with open(
                log_file,
                "a",
                encoding="utf-8",
                errors="replace"
            ) as f:
                if process.stdout is not None:
                    for line in process.stdout:
                        f.write(line)
                        f.flush()

                        if self.on_line:
                            self.on_line(line)

        threading.Thread(target=pump, daemon=True).start()

        log(self.name, "started (PID:", process.pid, ")")

    def alive(self):
        return self.p is not None and self.p.poll() is None

    def stop(self):
        if self.alive():
            log("Stopping", self.name)

            self.p.terminate()

            try:
                self.p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()

            log(self.name, "stopped")


# --------------------------------------------------
# PUBLISH TUNNEL URL
# --------------------------------------------------

def publish(url, settings):
    """Saves the tunnel URL and updates the Vercel UI."""

    (DATA / "backend_link.txt").write_text(
        f"API base URL: {url}/v1\n"
        f"API documentation: {url}/docs\n"
        f"Local UI: http://127.0.0.1:7860/ui\n"
        f"Time: {time.ctime()}\n",
        encoding="utf-8"
    )

    log("SERVER READY")
    log("Tunnel URL:", url)
    log("API:", url + "/v1")

    if settings["UPDATE_VERCEL"] != "1":
        return

    vercel = shutil.which("vercel")

    if not vercel:
        log("Vercel CLI not found. Automatic UI update skipped.")
        return

    try:
        build_script = HYBRID / "3_ui" / "build.py"
        ui_directory = HYBRID / "3_ui" / "medical-ai"

        if not build_script.exists():
            log("Build script not found:", build_script)
            return

        subprocess.run(
            [
                sys.executable,
                str(build_script),
                "--api",
                url
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=NO_WINDOW
        )

        if not ui_directory.exists():
            log("Vercel UI directory not found:", ui_directory)
            return

        output = subprocess.run(
            [vercel, "deploy", "--prod", "--yes"],
            cwd=ui_directory,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=NO_WINDOW
        )

        if output.returncode == 0:
            log("Vercel UI successfully updated.")
            log(output.stdout[-1000:])
        else:
            log(
                "Vercel deployment error:",
                output.stdout[-500:],
                output.stderr[-500:]
            )

    except Exception as e:
        log("Vercel update failed:", repr(e))


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--admin",
        help="Admin email address. Multiple emails can be separated by commas."
    )

    parser.add_argument(
        "--no-tunnel",
        action="store_true",
        help="Run locally without Cloudflare Tunnel."
    )

    args = parser.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)

    settings = load_settings(args.admin)

    if not settings["ADMIN_EMAILS"]:
        sys.exit(
            "First run: python hybrid/5_runner/run.py --admin your@email.com"
        )

    # --------------------------------------------------
    # CHECK REQUIRED FILES
    # --------------------------------------------------

    required_files = [
        MODEL_1,
        MODEL_2,
        LLAMA_SERVER
    ]

    if not args.no_tunnel:
        required_files.append(CLOUDFLARED)

    missing = [file for file in required_files if not file.is_file()]

    if missing:
        for file in missing:
            print(f"File not found: {file}")

        sys.exit("Required files are missing. Check the paths above.")

    # --------------------------------------------------
    # CHECK PORTS
    # --------------------------------------------------

    ports = [7860, 8080, 8081]

    for port in ports:
        if port_busy(port):
            sys.exit(
                f"Port {port} is already in use. "
                "Check whether another server is running."
            )

    # --------------------------------------------------
    # ENVIRONMENT
    # --------------------------------------------------

    env = {
        **os.environ,

        "ENGINE_URL": "http://127.0.0.1:8080",

        # Kept for compatibility with the adapted Gateway.
        # This points to the second SmolLM2 engine.
        "SMOLLM2_1_7B_ENGINE_URL": "http://127.0.0.1:8081",

        "DB_PATH": str(DATA / "personal_health_assistant.db"),

        "PYTHONIOENCODING": "utf-8",

        **{
            key: settings[key]
            for key in (
                "ADMIN_EMAILS",
                "SIGNUP_CODE",
                "RATE_LIMIT",
                "DAILY_LIMIT"
            )
        }
    }

    parallel = int(settings["PARALLEL"])
    threads = settings["THREADS"]

    # --------------------------------------------------
    # MODEL 1: SmolLM2-360M + SFT-QLoRA
    # --------------------------------------------------

    model_1 = Proc(
        "model_360m",
        [
            str(LLAMA_SERVER),
            "-m", str(MODEL_1),
            "--host", "127.0.0.1",
            "--port", "8080",
            "--ctx-size", str(parallel * 1024),
            "--parallel", str(parallel),
            "--threads", threads,
            "--jinja"
        ]
    )

    # --------------------------------------------------
    # MODEL 2: SmolLM2-1.7B-Instruct
    # --------------------------------------------------

    model_2 = Proc(
        "model_1_7b",
        [
            str(LLAMA_SERVER),
            "-m", str(MODEL_2),
            "--host", "127.0.0.1",
            "--port", "8081",
            "--ctx-size", str(parallel * 4096),
            "--parallel", str(parallel),
            "--threads", threads,
            "--jinja"
        ]
    )

    # --------------------------------------------------
    # GATEWAY
    # --------------------------------------------------

    gateway = Proc(
        "gateway",
        [
            sys.executable,
            "-m", "uvicorn",
            "gateway:app",
            "--host", "127.0.0.1",
            "--port", "7860"
        ],
        cwd=HYBRID / "1_backend",
        env=env
    )

    # --------------------------------------------------
    # CLOUDFLARE TUNNEL
    # --------------------------------------------------

    found = {}

    def on_tunnel_line(line):
        match = re.search(
            r"https://[a-z0-9-]+\.trycloudflare\.com",
            line
        )

        if match and found.get("url") != match.group(0):
            found["url"] = match.group(0)

    tunnel = Proc(
        "tunnel",
        [
            str(CLOUDFLARED),
            "tunnel",
            "--no-autoupdate",
            "--url",
            "http://127.0.0.1:7860"
        ],
        on_line=on_tunnel_line
    )

    # --------------------------------------------------
    # PROCESS LIST
    # --------------------------------------------------

    engines = [model_1, model_2]
    procs = engines + [gateway]

    if not args.no_tunnel:
        procs.append(tunnel)

    # --------------------------------------------------
    # START SERVICES
    # --------------------------------------------------

    log(
        "=== Personal Health Assistant ===",
        "| Admin:", settings["ADMIN_EMAILS"],
        "| Parallel:", parallel,
        "| Threads:", threads
    )

    try:
        for engine, port in zip(engines, (8080, 8081)):

            engine.start()

            if not wait_http(
                f"http://127.0.0.1:{port}/health",
                180
            ):
                log(
                    engine.name,
                    "did not become ready within 3 minutes. "
                    f"Check data/{engine.name}.log"
                )

                if not engine.alive():
                    raise RuntimeError(
                        f"{engine.name} exited during startup."
                    )

        gateway.start()

        if not wait_http(
            "http://127.0.0.1:7860/",
            30
        ):
            log("Gateway did not respond. Check data/gateway.log")

            if not gateway.alive():
                raise RuntimeError("Gateway exited during startup.")

        if not args.no_tunnel:
            tunnel.start()
        else:
            log("LOCAL MODE READY: http://127.0.0.1:7860/ui")

        # --------------------------------------------------
        # MONITOR AND RESTART PROCESSES
        # --------------------------------------------------

        published = None
        fails = 0
        tick = 0

        while True:

            # Publish a new Cloudflare URL.
            if found.get("url") and found["url"] != published:

                published = found["url"]
                fails = 0

                wait_http(published + "/", 60)

                publish(published, settings)

            tick += 1

            # Check public tunnel availability.
            if (
                published
                and not args.no_tunnel
                and tick % 4 == 0
            ):

                fails = (
                    0
                    if wait_http(published + "/", 1)
                    else fails + 1
                )

                if fails >= 3:

                    log(
                        "Tunnel is unreachable. "
                        "Restarting Cloudflare Tunnel."
                    )

                    tunnel.stop()
                    fails = 0

            # Restart stopped processes.
            for proc in procs:

                if not proc.alive():

                    log(proc.name, "stopped. Restarting...")

                    if proc is tunnel:
                        found.pop("url", None)

                    proc.start()

            time.sleep(15)

    except KeyboardInterrupt:
        log("Shutting down...")

    finally:
        for proc in reversed(procs):
            proc.stop()

        log("Personal Health Assistant stopped.")


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()
