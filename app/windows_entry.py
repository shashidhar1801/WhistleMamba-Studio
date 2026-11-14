"""Private Windows launcher bridge. No dependency installation or network access."""
import json
import sys
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from http.server import ThreadingHTTPServer


def serve(checkpoint, port):
    from whistlemamba_studio.web import Handler, Workspace
    from whistlemamba_studio.preview import VIEW_SECONDS

    build_id = Path(__file__).with_name("BUILD-ID").read_text().strip()

    class WindowsHandler(Handler):
        def handle_api(self, post):
            if not post and self.path == "/launcher-health" and self.headers.get("Host") == f"127.0.0.1:{port}":
                return self.j({"studio_build":build_id,"model":self.server.workspace.ready,
                               "preview_seconds":VIEW_SECONDS,"service":"WhistleMamba Studio for Windows"})
            return super().handle_api(post)

    with TemporaryDirectory(prefix="whistlemamba-windows-") as directory:
        with ThreadingHTTPServer(("127.0.0.1", port), WindowsHandler) as http:
            http.workspace = Workspace(directory, checkpoint)

            def control():
                # Private pipe inherited from the Windows owner, NOT a web stop endpoint.
                for command in sys.stdin:
                    if command.strip() != "STOP":
                        continue
                    with http.workspace.lock:
                        if any(row["phase"] == "running" for row in http.workspace.rows.values()):
                            print("STUDIO_BUSY", flush=True)
                            continue
                        http.shutdown()
                    return
                # Windows owner went away: let any in-flight detection finish.
                with http.workspace.run_lock:
                    http.shutdown()
            threading.Thread(target=control,daemon=True).start()
            print("STUDIO_READY",flush=True)
            http.serve_forever(poll_interval=.25)


if __name__ == "__main__":
    # Fail early on a missing/wrong asset instead of offering an unusable Run button.
    import hashlib
    from whistlemamba.registry import IDENTITY
    mode, ckpt = sys.argv[1:3]
    path = Path(ckpt)
    if not path.is_file():
        raise FileNotFoundError("The configured local CSR01 model file is missing.")
    if path.stat().st_size > 100_000_000 or hashlib.sha256(path.read_bytes()).hexdigest()!=IDENTITY["checkpoint_sha256"]:
        raise ValueError("Configured checkpoint is not the verified CSR01 inference asset.")
    if mode == "web":
        serve(ckpt,int(sys.argv[3]))
    elif mode == "desktop":
        from whistlemamba_studio.application import main
        raise SystemExit(main(["--checkpoint",ckpt]))
    elif mode == "check-editor":
        from PySide6.QtWidgets import QApplication
        from whistlemamba_studio.main_window import MainWindow
        app=QApplication([]);win=MainWindow(initial_checkpoint=ckpt);win.show();app.processEvents()
        assert win.home.open_button.isVisible() and win.view.viewport_seconds==15
        win.close()
    else:
        raise ValueError("Unknown launcher mode.")
