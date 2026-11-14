"""Windows integration check (manual, synthetic only). Runs its own loopback service.

Usage: python manual_browser_exit.py --build PATH --fixture synthetic_whistles_4s.wav
       --out NEW_DIRECTORY [--chrome PATH]
Requires local Playwright, Pillow, Chrome, pdfinfo and pdftoppm in PATH,
and the Windows build's already authorized WSL runtime/model configuration.
"""
import argparse
import csv
import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import threading
import time
import urllib.request
import wave
import zipfile

from PIL import Image
from playwright.sync_api import sync_playwright


def sha(b):
    return hashlib.sha256(b).hexdigest()


def linux(p):
    p = Path(p).resolve().as_posix()
    if p[1:3] != ":/":
        raise ValueError("Use a local drive")
    return "/mnt/" + p[0].lower() + p[2:]


def health(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/launcher-health", timeout=2) as r:
        return json.load(r)


def consoles():
    seen = set()
    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, unused):
        if ctypes.windll.user32.IsWindowVisible(hwnd):
            name = ctypes.create_unicode_buffer(256)
            ctypes.windll.user32.GetClassNameW(hwnd, name, len(name))
            if name.value in ("ConsoleWindowClass", "CASCADIA_HOSTING_WINDOW_CLASS",
                              "PseudoConsoleWindow"):
                seen.add(int(hwnd))
        return True
    ctypes.windll.user32.EnumWindows(visit, 0)
    return seen


def check_zip(path, wav, expected, directory):
    names = {"export_manifest.json", "contours.csv", "raven_selections.txt",
             "analysis.wmproject.json", "reproducibility.json",
             "spectrogram_overlay.png", "spectrogram_overlay.pdf"}
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None and len(z.namelist()) == 7 and set(z.namelist()) == names
        z.extractall(directory)
    load = lambda n: json.loads((directory/n).read_bytes())
    manifest = load("export_manifest.json"); members = set()
    for v in manifest.values():
        leaf = Path(v["path"]).name
        assert leaf in names and sha((directory/leaf).read_bytes()) == v["sha256"]
        members.add(leaf)
    assert members == names-{"export_manifest.json"} and len(manifest) == 6
    project = load("analysis.wmproject.json"); r = load("reproducibility.json")
    assert project["analysis_complete"] and r["audio_hash_verified"]
    assert sha(wav.read_bytes()) == r["audio"]["sha256"] == project["audio"]["sha256"]
    assert project["model"]["checkpoint_sha256"] == expected["checkpoint_sha256"]
    assert project["model"]["model_id"] == r["model"]["model_id"] == expected["model_id"]
    tracks = project["edited_contours"]
    with (directory/"contours.csv").open(encoding="utf8", newline="") as f:
        points = list(csv.DictReader(f))
    with (directory/"raven_selections.txt").open(encoding="utf8", newline="") as f:
        events = list(csv.DictReader(f, delimiter="\t"))
    assert len(points) == sum(len(c["points"]) for c in tracks)
    assert len(events) == len(tracks) == r["edited_contour_count"] == r["original_contour_count"]
    with Image.open(directory/"spectrogram_overlay.png") as im:
        im.load(); assert im.width >= 1000 and im.height >= 500
        assert min(im.convert("L").getextrema()) < 100
        image_size = list(im.size)
    flags = dict(capture_output=True, check=True, timeout=25, creationflags=subprocess.CREATE_NO_WINDOW)
    pdf = directory/"spectrogram_overlay.pdf"
    info = subprocess.run(["pdfinfo", str(pdf)], **flags)
    assert re.search(rb"Pages:\s+1\b", info.stdout)
    subprocess.run(["pdftoppm","-f","1","-l","1","-singlefile","-r","65","-png",str(pdf),str(directory/"page-check")], **flags)
    with Image.open(directory/"page-check.png") as im:
        im.load(); lo, hi = im.convert("L").getextrema(); assert hi-lo > 100
    return dict(archive_sha256=sha(path.read_bytes()), members=7, payload_hashes_verified=6,
                input_hash_verified=True, tracks=len(tracks), points=len(points),
                decoded_png_size=image_size, pdf_pages_rendered=1, source_rate_hz=r["audio"]["sample_rate_hz"])


# Obtain the actual Linux child's return code, independently of the Windows WSL
# wrapper's status. Keep the child's stdin connected to the launcher's STOP pipe.
SUPERVISE = """import subprocess,sys
p=subprocess.Popen([sys.executable,'-u','windows_entry.py','web',*sys.argv[1:]])
rc=p.wait()
print('STUDIO_NATIVE_EXIT',rc,flush=True)
sys.exit(rc if rc>=0 else 128-rc)
"""

def cycle(browser, root, cfg, wav, folder, identity, build_id):
    folder.mkdir()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0));port=sock.getsockname()[1]
    assert port != 8785
    url=f"http://127.0.0.1:{port}"
    context=browser.new_context(accept_downloads=True)
    page=context.new_page()
    # WSL forwards stdout and stderr independently; a shared output handle on
    # Windows can interleave/overwrite the native-exit marker with warnings.
    logpath=folder/"service.out.log"; errpath=folder/"service.err.log"
    logfile=logpath.open("wb"); errfile=errpath.open("wb")
    args=["wsl.exe","-d",cfg["distribution"],"--cd",linux(root/"app"),"--exec",
          "/usr/bin/env","PYTHONPATH=.","PYTHONDONTWRITEBYTECODE=1","PYTHONFAULTHANDLER=1",
          cfg["linux_python"],"-u","-c",SUPERVISE,linux(cfg["checkpoint_windows"]),str(port)]
    proc=subprocess.Popen(args, creationflags=subprocess.CREATE_NO_WINDOW, stdin=subprocess.PIPE,
                          stdout=logfile, stderr=errfile)
    try:
        for _ in range(120):
            try: status=health(port);break
            except OSError:
                assert proc.poll() is None, "service stopped during startup"
                time.sleep(.25)
        else:raise TimeoutError("service startup")
        assert status["studio_build"]==build_id and status["model"]
        page.goto(url)
        def until(js,seconds=60):
            deadline=time.monotonic()+seconds
            while time.monotonic()<deadline:
                if page.evaluate(js):return
                page.wait_for_timeout(150)
            raise TimeoutError(js)
        until("Boolean(token)")
        assert page.evaluate("records.size") == 0
        page.locator("#files").set_input_files(wav)
        until("Boolean(current) && !document.querySelector('#run').disabled")
        page.locator("#run").click()
        until("['done','error','failed'].includes(records.get(current).phase)",180)
        assert page.evaluate("records.get(current).phase")=="done"
        with page.expect_download(timeout=90000) as event:
            page.locator("#save").click()
        dest=folder/"review.zip";event.value.save_as(dest)
        assert event.value.failure() is None and dest.stat().st_size>1000
        start=time.monotonic()
        proc.stdin.write(b"STOP\n");proc.stdin.flush()
        wrapper_rc=proc.wait(timeout=40)
        elapsed=time.monotonic()-start
    finally:
        proc.stdin.close()
        try:proc.wait(timeout=150)
        finally:logfile.close();errfile.close();context.close()
    log=logpath.read_text(errors="replace")+"\n"+errpath.read_text(errors="replace")
    native=re.findall(r"(?m)^STUDIO_NATIVE_EXIT (-?\d+)\s*$",log)
    assert native==["0"] and wrapper_rc==0,(native,wrapper_rc)
    assert not re.search(r"segfault|segmentation fault|fatal python error|traceback|outside of the main thread|QObject::",log,re.I), "unexpected diagnostic"
    with socket.socket() as sock:
        sock.settimeout(1);assert sock.connect_ex(("127.0.0.1",port))!=0,"listener still alive"
    result=check_zip(dest,wav,identity,folder/"unzipped")
    result.update(native_exit=int(native[0]),wsl_exit=wrapper_rc,shutdown_seconds=round(elapsed,3),
                  port_closed=True,export_or_shutdown_error_in_log=False)
    print(f"{wav.name}: native exit 0; archive re-opened and verified after shutdown",flush=True)
    return result


def main():
    a=argparse.ArgumentParser(description=__doc__)
    for k in ("build","fixture","out"):a.add_argument("--"+k,type=Path,required=True)
    a.add_argument("--chrome",type=Path,default=Path(os.environ["ProgramFiles"])/"Google/Chrome/Application/chrome.exe")
    opt=a.parse_args();out=opt.out.resolve();out.mkdir(exist_ok=False,parents=True)
    root=opt.build.resolve();cfg=json.loads((root/"LocalRuntime.json").read_text())
    identity=json.loads((root/"app/whistlemamba/resources/csr01_identity.json").read_bytes())
    build=(root/"app/BUILD-ID").read_text().strip()
    fixture=opt.fixture.resolve()
    assert sha(fixture.read_bytes())=="d45a91bd8a38be680d7cd91f92ab1834d7cab8b9cb85abc06c34f2c8a721eb54"
    limited=out/"synthetic-96k.wav"
    with wave.open(str(limited),"wb") as f:
        f.setparams((1,2,96000,0,"NONE","not compressed"))
        samples=(int(8000*math.sin(2*math.pi*(7000*(i/96000)+350*(i/96000)**2))) for i in range(4*96000))
        f.writeframes(b"".join(struct.pack("<h",x) for x in samples))
    before=consoles();new=set();end=threading.Event()
    def observe():
        while not end.is_set():
            new.update(consoles()-before);end.wait(.05)
    monitor=threading.Thread(target=observe);monitor.start()
    try:
        with sync_playwright() as pw:
            b=pw.chromium.launch(executable_path=str(opt.chrome),headless=True,args=["--mute-audio"])
            try:
                results=[cycle(b,root,cfg,wav,out/name,identity,build) for wav,name in
                         [(limited,"96k"),(fixture,"192k")]]
            finally:b.close()
    finally:end.set();monitor.join()
    assert not new, "new visible console observed"
    receipt=dict(build_id=build,fixture="synthetic only; one real model analysis per isolated service",
                 protocol="Chrome download; launcher STOP via stdin; native child wait; reopen ZIP after exit",
                 new_visible_console_windows_observed=len(new),runs=results)
    (out/"verified.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("PASS: two clean isolated sessions",flush=True)

if __name__=="__main__":main()
