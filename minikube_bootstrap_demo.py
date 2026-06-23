#!/usr/bin/env python3
"""
minikube_bootstrap_demo.py

Best-effort bootstrap demo for Ubuntu/LXC-style environments.

What it does:
- Serves a local web UI on port 9880
- Streams live logs in the browser
- Attempts to prepare:
  - CNI plugins
  - crictl
  - cri-dockerd
  - minikube startup with none driver + docker runtime

Run:
  sudo python3 minikube_bootstrap_demo.py

Open:
  http://127.0.0.1:9880
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Optional

PORT = 9880
LOG_LIMIT = 4000

CNI_VERSION = "v1.8.0"
CRI_TOOLS_VERSION = "v1.35.0"
CRI_DOCKERD_VERSION = "v0.4.0"

CNI_URL = f"https://github.com/containernetworking/plugins/releases/download/{CNI_VERSION}/cni-plugins-linux-amd64-{CNI_VERSION}.tgz"
CRICTL_URL = f"https://github.com/kubernetes-sigs/cri-tools/releases/download/{CRI_TOOLS_VERSION}/crictl-{CRI_TOOLS_VERSION.lstrip('v')}-linux-amd64.tar.gz"
CRI_DOCKERD_URL = f"https://github.com/Mirantis/cri-dockerd/releases/download/{CRI_DOCKERD_VERSION}/cri-dockerd-{CRI_DOCKERD_VERSION.lstrip('v')}.amd64.tgz"
MINIKUBE_URL = "https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64"

CNI_DIR = Path("/opt/cni/bin")
WORKDIR = Path("/root/minikube-bootstrap")
CRICTL_PATH = Path("/usr/local/bin/crictl")
CRI_DOCKERD_PATH = Path("/usr/local/bin/cri-dockerd")
CRI_DOCKERD_SERVICE = Path("/etc/systemd/system/cri-docker.service")


class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.logs: List[str] = []
        self.running = False
        self.done = False
        self.failed = False
        self.current = "idle"
        self.commands: List[str] = []
        self.exit_code: Optional[int] = None

    def log(self, message: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        with self.lock:
            self.logs.append(line)
            if len(self.logs) > LOG_LIMIT:
                self.logs = self.logs[-LOG_LIMIT:]
        print(line, flush=True)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "running": self.running,
                "done": self.done,
                "failed": self.failed,
                "current": self.current,
                "exit_code": self.exit_code,
                "commands": self.commands[-50:],
                "logs": self.logs[-400:],
            }


STATE = State()


def is_root() -> bool:
    return os.geteuid() == 0


def have_cmd(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run(cmd: List[str], check: bool = True, env: Optional[dict] = None) -> subprocess.CompletedProcess:
    STATE.log(f"$ {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    out = proc.stdout or ""
    if out.strip():
        for line in out.rstrip().splitlines():
            STATE.log(line)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=out)
    return proc


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    STATE.log(f"Downloading {url}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    STATE.log(f"Saved to {dest}")


def extract_tar(tar_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    run(["tar", "-xzf", str(tar_path), "-C", str(target_dir)])


def install_packages() -> None:
    env = os.environ.copy()
    env["DEBIAN_FRONTEND"] = "noninteractive"
    run(["apt", "update"], env=env)
    run([
        "apt", "install", "-y",
        "curl", "wget", "tar", "ca-certificates",
        "conntrack", "socat", "ebtables", "ethtool",
        "iproute2", "iptables", "jq"
    ], check=False, env=env)


def ensure_cni() -> None:
    CNI_DIR.mkdir(parents=True, exist_ok=True)
    bridge = CNI_DIR / "bridge"
    host_local = CNI_DIR / "host-local"
    loopback = CNI_DIR / "loopback"

    if bridge.exists() and host_local.exists() and loopback.exists():
        STATE.log("CNI plugins already present.")
        return

    WORKDIR.mkdir(parents=True, exist_ok=True)
    tgz = WORKDIR / "cni-plugins.tgz"
    download(CNI_URL, tgz)
    extract_tar(tgz, CNI_DIR)
    for name in ("bridge", "host-local", "loopback"):
        p = CNI_DIR / name
        if p.exists():
            p.chmod(0o755)
    STATE.log(f"CNI plugins installed in {CNI_DIR}")


def ensure_crictl() -> None:
    if have_cmd("crictl"):
        STATE.log("crictl already available.")
        return

    WORKDIR.mkdir(parents=True, exist_ok=True)
    tgz = WORKDIR / "crictl.tgz"
    download(CRICTL_URL, tgz)
    run(["tar", "-xzf", str(tgz), "-C", "/usr/local/bin"])
    CRICTL_PATH.chmod(0o755)
    STATE.log("crictl installed.")


def ensure_cri_dockerd() -> None:
    if have_cmd("cri-dockerd"):
        STATE.log("cri-dockerd already available.")
    else:
        WORKDIR.mkdir(parents=True, exist_ok=True)
        tgz = WORKDIR / "cri-dockerd.tgz"
        download(CRI_DOCKERD_URL, tgz)
        extract_dir = WORKDIR / "cri-dockerd-extract"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        run(["tar", "-xzf", str(tgz), "-C", str(extract_dir)])
        bin_path = None
        for candidate in extract_dir.rglob("cri-dockerd"):
            if candidate.is_file():
                bin_path = candidate
                break
        if not bin_path:
            raise FileNotFoundError("cri-dockerd binary not found after extraction")
        shutil.copy2(bin_path, CRI_DOCKERD_PATH)
        CRI_DOCKERD_PATH.chmod(0o755)
        STATE.log("cri-dockerd binary installed.")

    if not CRI_DOCKERD_SERVICE.exists():
        CRI_DOCKERD_SERVICE.write_text(
            """[Unit]
Description=CRI Interface for Docker
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service

[Service]
Type=simple
ExecStart=/usr/local/bin/cri-dockerd
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
"""
        )
        STATE.log("cri-dockerd systemd service created.")

    run(["systemctl", "daemon-reload"], check=False)
    run(["systemctl", "enable", "--now", "cri-docker"], check=False)
    run(["systemctl", "status", "cri-docker", "--no-pager", "-l"], check=False)


def ensure_minikube() -> None:
    if have_cmd("minikube"):
        STATE.log("minikube already available.")
        return

    WORKDIR.mkdir(parents=True, exist_ok=True)
    dest = WORKDIR / "minikube-linux-amd64"
    download(MINIKUBE_URL, dest)
    dest.chmod(0o755)
    shutil.copy2(dest, "/usr/local/bin/minikube")
    Path("/usr/local/bin/minikube").chmod(0o755)
    STATE.log("minikube installed.")

def reset_minikube_state() -> None:
    run(["minikube", "delete"], check=False)
    for path in [
        Path("/root/.minikube"),
        Path("/root/.kube"),
        Path("/etc/kubernetes"),
        Path("/var/lib/etcd"),
    ]:
        if path.exists():
            STATE.log(f"Removing {path}")
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass


def start_minikube() -> None:
    cmd = [
        "minikube", "start",
        "--driver=none",
        "--container-runtime=docker",
        "--cni=bridge",
        "--force",
    ]
    STATE.commands.append(" ".join(cmd))
    run(cmd, check=False)


def bootstrap() -> None:
    STATE.running = True
    STATE.current = "starting"
    try:
        if not is_root():
            STATE.log("Run this script as root.")
            STATE.failed = True
            return

        STATE.current = "installing packages"
        install_packages()

        STATE.current = "installing CNI"
        ensure_cni()

        STATE.current = "installing crictl"
        ensure_crictl()

        STATE.current = "installing cri-dockerd"
        ensure_cri_dockerd()

        STATE.current = "installing minikube"
        ensure_minikube()

        STATE.current = "cleaning minikube state"
        reset_minikube_state()

        STATE.current = "starting minikube"
        start_minikube()

        STATE.current = "finished"
        STATE.done = True
        STATE.log("Bootstrap sequence completed.")
    except Exception as e:
        STATE.failed = True
        STATE.log(f"ERROR: {e}")
    finally:
        STATE.running = False
        if not STATE.done and not STATE.failed:
            STATE.failed = True


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Minikube Bootstrap Demo</title>
  <style>
    body { font-family: monospace; background:#111; color:#eee; margin:0; padding:16px; }
    .top { display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
    .badge { padding:4px 10px; border:1px solid #444; border-radius:999px; }
    pre { white-space:pre-wrap; background:#000; border:1px solid #333; padding:12px; height:75vh; overflow:auto; }
    button, a.btn { background:#1f2937; color:#fff; border:1px solid #444; padding:8px 12px; text-decoration:none; cursor:pointer; }
  </style>
</head>
<body>
  <div class="top">
    <div class="badge" id="state">loading...</div>
    <button onclick="location.reload()">refresh</button>
    <a class="btn" href="/start">start bootstrap</a>
  </div>
  <h3>Live output</h3>
  <pre id="log"></pre>
<script>
async function update() {
  const r = await fetch('/status');
  const s = await r.json();
  document.getElementById('state').textContent =
    `running=${s.running} done=${s.done} failed=${s.failed} current=${s.current} exit=${s.exit_code}`;
  document.getElementById('log').textContent = s.logs.join('\\n');
  document.getElementById('log').scrollTop = document.getElementById('log').scrollHeight;
}
setInterval(update, 1000);
update();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, content_type: str = "text/plain; charset=utf-8") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            self._send(200, HTML.encode(), "text/html; charset=utf-8")
            return
        if self.path == "/status":
            body = json.dumps(STATE.snapshot(), indent=2).encode()
            self._send(200, body, "application/json")
            return
        if self.path == "/start":
            if not STATE.running:
                threading.Thread(target=bootstrap, daemon=True).start()
            self.send_response(302)
            self.send_header("Location", "/")
            self.end_headers()
            return
        self._send(404, b"not found")


def main() -> None:
    STATE.log(f"Opening web UI on http://127.0.0.1:{PORT}")
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    if os.geteuid() != 0:
        STATE.log("WARNING: run as root for the install/start steps.")
    threading.Thread(target=bootstrap, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        STATE.log("Shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
