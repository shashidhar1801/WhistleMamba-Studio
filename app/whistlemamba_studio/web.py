"""Loopback-only reviewer workspace. No web service, telemetry or external assets."""
from __future__ import annotations
import argparse, hashlib, io, json, logging, os, re, secrets, shutil, tempfile, threading, zipfile
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, unquote, parse_qs

import soundfile as sf
from whistlemamba.project import AnalysisProject, ContourPoint, Contour
from .web_timeline import tile, display_palettes
from math import isfinite
from whistlemamba.input_mode import LIMITED_NOTICE, rate_supported
from whistlemamba.editing import ContourEditor
from whistlemamba.registry import IDENTITY, validate_project_identity
from whistlemamba.spectrogram import SpectrogramSettings
from whistlemamba_studio.preview import VIEW_SECONDS, display_tile, viewport

LOG=logging.getLogger("studio.web")
LIMIT=512*1024**2
STATIC=Path(__file__).parent/"web_assets"

def completion_summary(mode, result):
    def count(n, label):
        return f"{n} {label} candidate track{'s' if n != 1 else ''}"
    if mode == "learned":
        return count(result.learned_contours, "learned")
    if mode == "heuristic":
        return count(result.heuristic_contours, "heuristic")
    if mode == "both":
        learned, heuristic = result.learned_contours, result.heuristic_contours
        total = learned + heuristic
        return f"{learned} learned + {heuristic} heuristic = {total} candidate track{'s' if total != 1 else ''}"
    raise ValueError("Unknown tracker mode.")


class Workspace:
    def __init__(self, root, checkpoint):
        self.root=Path(root); self.rows={}; self.checkpoint=checkpoint
        self.ready=bool(checkpoint and Path(checkpoint).is_file())
        self.token=secrets.token_urlsafe(32); self.lock=threading.RLock()
        self.run_lock=threading.Lock(); self.predictor=None; self.reserved=0
        self.tile_slots=threading.BoundedSemaphore(2)

    def receive(self, stream, size, name):
        if not (0 < size <= LIMIT): raise ValueError("Choose a WAV smaller than 512 MiB and not empty.")
        if not name.lower().endswith(".wav"): raise ValueError("Choose a .wav file.")
        with self.lock:
            if len(self.rows)>=64 or self.reserved+size>2*1024**3: raise ValueError("Session limit reached. Remove a recording first.")
            self.reserved+=size
        key=secrets.token_hex(12); base=self.root/key
        try:
            base.mkdir()
            name=re.sub(r'[^\w .()+-]','_',name.replace('\\','/').rsplit('/',1)[-1])[-150:]
            path=base/name
            with path.open('xb') as dst:
                left=size
                while left:
                    chunk=stream.read(min(left,1024**2))
                    if not chunk:raise ValueError("File transfer interrupted.")
                    dst.write(chunk);left-=len(chunk)
            info=sf.info(path)
            if info.format not in {"WAV","WAVEX","RF64"} or info.frames<2:raise ValueError("Not a readable WAV recording.")
            if info.channels>64 or info.samplerate>768000:raise ValueError("This browser preview supports up to 64 channels and 768 kHz.")
            project=AnalysisProject.create(path)
            with self.lock:
                self.rows[key]=dict(key=key,name=name,path=path,size=size,project=project,editor=None,phase="ready",progress="",image=None,stop=threading.Event(),readers=0,removed=False)
            return key
        except Exception:
            with self.lock:self.reserved-=size
            if base.parent==self.root and base.is_dir():shutil.rmtree(base)
            raise

    def render_tile(self,k,index):
        # Tile reads may span several file opens. An old request may finish
        # after the user removes/switches a recording. Release its copy only
        # when the last in-flight reader is done.
        with self.lock:
            row=self.rows[k]
            row["readers"]+=1
        try:
            with self.tile_slots:
                return tile(row,index)
        finally:
            with self.lock:
                row["readers"]-=1
                if row["removed"] and not row["readers"]:
                    shutil.rmtree(row["path"].parent)
                    self.reserved-=row["size"]

    def remove(self,k):
        with self.lock:
            r=self.rows[k]
            if r["phase"]=="running":raise ValueError("Wait for analysis before removing this copy.")
            r["removed"]=True
            del self.rows[k]
            if not r["readers"]:
                shutil.rmtree(r["path"].parent)
                self.reserved-=r["size"]

    def snapshot(self,k):
        with self.lock:
            r=self.rows[k]; p=r["project"]; a=p.audio
            return dict(id=k,name=r["name"],seconds=a.duration_s,rate=a.sample_rate_hz,channels=a.channels,
                phase=r["phase"],progress=r["progress"],compatible=a.duration_s>=4 and rate_supported(a.sample_rate_hz),
                mode_notice=LIMITED_NOTICE if a.sample_rate_hz==96000 else "",
                options=r.get("options"),undo=bool(r["editor"] and r["editor"].can_undo),redo=bool(r["editor"] and r["editor"].can_redo),
                tracks=[dict(id=c.contour_id,state=c.review_state,source=c.source,notes=c.notes,n=len(c.points),
                    start=c.points[0].time_s,end=c.points[-1].time_s) for c in p.edited_contours])
    def analyze(self,k,tracker="learned",batch=1):
        if not self.ready:raise ValueError("Start Studio with the authorized local model to enable analysis.")
        if not self.snapshot(k)["compatible"]:raise ValueError("Use at least 4 s at 96 kHz (experimental) or 192 kHz and above.")
        if tracker not in ("learned","both","heuristic") or type(batch) is not int or not 1<=batch<=16:
            raise ValueError("Invalid tracker or batch size.")
        if not self.run_lock.acquire(False):raise ValueError("Another recording is being analyzed. Wait for it to finish.")
        r=self.rows[k]
        if r["phase"]=="done":
            self.run_lock.release();raise ValueError("This recording has already been analyzed.")
        r["options"]=dict(tracker=tracker,batch=batch);r["stop"].clear();r["phase"]="running";r["progress"]="Loading the local model…"
        def run():
            try:
                from whistlemamba.inference import WhistleMambaPredictor, InferenceSettings, InferenceCancelled
                settings=InferenceSettings(device="cuda",tracker_mode=tracker,batch_size=batch,presence_gating=False)
                if self.predictor is None or self.predictor.settings != settings:
                    self.predictor=None
                    self.predictor=WhistleMambaPredictor(checkpoint_path=self.checkpoint,settings=settings)
                def progress(e):
                    with self.lock:r["progress"]=f"Window {e.completed_windows} of {e.total_windows}"
                try:
                    result=self.predictor.analyze_file(r["path"],progress=progress,cancelled=r["stop"].is_set,
                        resume_path=r["path"].parent/f"resume-{tracker}-{batch}.json")
                except InferenceCancelled:
                    with self.lock:r.update(phase="cancelled",progress="Cancelled. Run again with the same settings to resume.")
                    return
                validate_project_identity(result.project.model)
                with self.lock:r.update(project=result.project, editor=ContourEditor(result.project), phase="done", progress=completion_summary(tracker, result))
            except Exception:
                LOG.exception("Inference failed")
                with self.lock:r.update(phase="failed",progress="Analysis failed. Check local GPU memory, model and server log.")
            finally:self.run_lock.release()
        threading.Thread(target=run,daemon=True).start()

    def shapes(self,k,start,end):
        p=self.rows[k]["project"]
        if not all(isinstance(x,(int,float)) and isfinite(x) for x in (start,end)) or not 0<=start<=end<=start+65:
            raise ValueError("Invalid time range.")
        return [dict(id=c.contour_id,source=c.source,state=c.review_state,
                     points=[[q.time_s,q.frequency_hz] for q in c.points])
                for c in p.edited_contours if c.points[-1].time_s>=start and c.points[0].time_s<=end]

    def edit(self,k,d):
        r=self.rows[k];e=r["editor"];p=r["project"]
        if r["phase"]!="done" or not e:raise ValueError("Analyze before editing.")
        op=d.get("op"); cid=d.get("id")
        if op in ("new","add","move"):
            t=float(d["time"]);f=float(d["frequency"])
            if not (isfinite(t) and isfinite(f) and 0<=t<=p.audio.duration_s and 0<=f<=p.audio.sample_rate_hz/2):
                raise ValueError("Point outside the recording.")
            point=ContourPoint(t,f)
        if op in ("move","drop-point","split"):
            c=next((c for c in p.edited_contours if c.contour_id==cid),None)
            i=d.get("index")
            if not c or type(i) is not int or not 0<=i<len(c.points):raise ValueError("Choose a valid point.")
        if op=="new":e.add_contour(Contour.create([point],source="manual"))
        elif op=="add":e.add_point(cid,point)
        elif op=="move":e.move_point(cid,i,point)
        elif op=="drop-point":e.delete_point(cid,i)
        elif op=="split":e.split(cid,i)
        elif op=="merge":
            if cid==d.get("other"):raise ValueError("Choose two different contours.")
            e.merge(cid,d["other"])
        elif op=="delete":e.delete_contour(cid)
        elif op=="notes":e.set_notes(cid,str(d.get("notes",""))[:4096])
        elif op=="state":e.set_review_state(cid,d["state"])
        elif op=="undo":e.undo()
        elif op=="redo":e.redo()
        else:raise ValueError("Unknown edit.")

    def plot(self,k,start):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib import rc_context
        r=self.rows[k];p=r["project"];a=p.audio
        start,span=viewport(a.duration_s,start)
        hi=min(96000.,a.sample_rate_hz/2);lo=4000. if hi>4100 else 0.
        tile=display_tile(r['path'],start,span,SpectrogramSettings(f_min_hz=lo,f_max_hz=hi))
        fig=Figure(figsize=(9.1,3.65),dpi=120,layout="constrained");FigureCanvasAgg(fig);ax=fig.subplots()
        ax.imshow(tile.magnitude_db,origin="lower",aspect="auto",interpolation="nearest",cmap="magma",vmin=tile.display_min_db,vmax=tile.display_max_db,
             extent=(start,start+span,(tile.frequency_hz[0]-.5*a.sample_rate_hz/2048)/1000,(tile.frequency_hz[-1]+.5*a.sample_rate_hz/2048)/1000))
        for c in p.edited_contours:
            if c.review_state=="rejected" or c.points[-1].time_s<start or c.points[0].time_s>start+span:continue
            ax.plot([pt.time_s for pt in c.points],[pt.frequency_hz/1000 for pt in c.points],lw=1.3,color='#59e9ff')
        ax.set(xlim=(start,start+span),ylim=(lo/1000,hi/1000),xlabel="Time (s)",ylabel="Frequency (kHz)")
        b=io.BytesIO();fig.savefig(b,format="png");fig.clear();return b.getvalue()

    def archive(self,k):
        from whistlemamba.export import export_bundle
        r=self.rows[k]
        if r["phase"]!="done": raise ValueError("Analyze this recording before exporting.")
        with tempfile.TemporaryDirectory(dir=r['path'].parent) as td:
            outputs=export_bundle(r['project'],td)
            b=io.BytesIO()
            with zipfile.ZipFile(b,"w",zipfile.ZIP_DEFLATED) as z:
                for f in outputs.values():z.write(f,Path(f).name)
            return b.getvalue()


class Handler(BaseHTTPRequestHandler):
    server_version="WhistleMambaStudio"
    def setup(self):
        super().setup();self.connection.settimeout(120)
    def reply(self,status,b,mime,extra=None):
        self.send_response(status);self.send_header("Content-Type",mime);self.send_header("Content-Length",str(len(b)))
        self.send_header("Cache-Control","no-store");self.send_header("X-Content-Type-Options","nosniff");self.send_header("Referrer-Policy","no-referrer")
        self.send_header("Content-Security-Policy","default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; media-src 'self'; connect-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'")
        for n,v in (extra or {}).items():self.send_header(n,v)
        self.end_headers();self.wfile.write(b)
    def j(self,v,code=200):self.reply(code,json.dumps(v).encode(),"application/json")
    def audio(self,k):
        w=self.server.workspace
        cookies=self.headers.get("Cookie","").split(";")
        if not any(secrets.compare_digest(c.strip(),"studio_media="+w.token) for c in cookies):
            return self.j({"error":"Reload Studio first."},403)
        with w.lock:
            if k not in w.rows:return self.j({"error":"Recording unavailable."},404)
            r=w.rows[k]; length=r["size"]; src=r["path"].open("rb")
        with src:
            first,last=0,length-1; rng=self.headers.get("Range")
            if rng:
                m=re.fullmatch(r"bytes=(\d*)-(\d*)",rng)
                if m and any(m.groups()):
                    a,b=m.groups()
                    if not a:first=max(0,length-int(b))
                    else:first=int(a);last=min(last,int(b)) if b else last
                else:first=length
                if first>last:
                    return self.reply(416,b"","audio/wav",{"Content-Range":f"bytes */{length}"})
            self.send_response(206 if rng else 200)
            self.send_header("Content-Type","audio/wav");self.send_header("Accept-Ranges","bytes")
            self.send_header("Cache-Control","no-store");self.send_header("Cross-Origin-Resource-Policy","same-origin")
            if rng:self.send_header("Content-Range",f"bytes {first}-{last}/{length}")
            left=last-first+1;self.send_header("Content-Length",str(left));self.end_headers();src.seek(first)
            while left:
                block=src.read(min(left,256*1024))
                if not block:break
                self.wfile.write(block);left-=len(block)
    def handle_api(self,post):
        host=self.headers.get("Host",""); port=self.server.server_port
        if host not in (f"127.0.0.1:{port}",f"localhost:{port}"):return self.j({"error":"Local access only."},403)
        path=urlsplit(self.path).path
        w=self.server.workspace
        if not post:
            if path in ("/","/studio.js","/timeline.js","/appearance.js","/studio.css"):
                f,m={"/":("index.html","text/html; charset=utf-8"),"/studio.js":("studio.js","text/javascript; charset=utf-8"),"/timeline.js":("timeline.js","text/javascript; charset=utf-8"),"/appearance.js":("appearance.js","text/javascript; charset=utf-8"),"/studio.css":("studio.css","text/css; charset=utf-8")}[path]
                return self.reply(200,(STATIC/f).read_bytes(),m)
            if path=="/session":
                with w.lock:body=json.dumps(dict(token=w.token,model=w.ready,previewSeconds=VIEW_SECONDS,displayPalettes=display_palettes(),recordings=[w.snapshot(i) for i in w.rows])).encode()
                return self.reply(200,body,"application/json",{"Set-Cookie":"studio_media="+w.token+"; Path=/media/; HttpOnly; SameSite=Strict"})
            media=re.fullmatch(r"/media/([0-9a-f]{24})",path)
            if media:return self.audio(media[1])
            return self.j({"error":"Page not found."},404)
        if self.headers.get("Origin") not in (None,"http://"+host) or not secrets.compare_digest(self.headers.get("X-Studio-Token",""),w.token):
            return self.j({"error":"Reload the local Studio page."},403)
        n=int(self.headers.get('Content-Length','0'))
        if path=="/recordings":
            key=w.receive(self.rfile,n,unquote(self.headers.get("X-Wav-Name","recording.wav")));return self.j(w.snapshot(key),201)
        if not 0<=n<=65536:raise ValueError("Request too large.")
        data=json.loads(self.rfile.read(n) or b'{}')
        if not isinstance(data,dict):raise ValueError("Invalid request.")
        match=re.fullmatch(r'/r/([0-9a-f]{24})/(state|plot|tile|shapes|run|cancel|edit|review|export|remove)',path)
        if not match:return self.j({"error":"Not found."},404)
        k,cmd=match.groups()
        if cmd=="tile":
            return self.j(w.render_tile(k,data.get("index")))
        with w.lock:
            if k not in w.rows:return self.j({"error":"This recording is no longer in the session."},404)
            r=w.rows[k]
            if cmd=="plot":return self.reply(200,w.plot(k,float(data.get('start',0))),"image/png")
            if cmd=="export":return self.reply(200,w.archive(k),"application/zip")
            if cmd=="shapes":return self.j(w.shapes(k,data.get("start"),data.get("end")))
            if cmd=="run":w.analyze(k,data.get("tracker","learned"),data.get("batch",1))
            if cmd=="cancel" and r["phase"]=="running":r["stop"].set();r["progress"]="Cancelling after the current batch…"
            if cmd=="edit":w.edit(k,data)
            if cmd=="remove":
                w.remove(k);return self.j({"removed":True})
            if cmd=="review":
                if r["phase"]!="done":raise ValueError("Analyze the recording first.")
                if data.get("state") not in ("pending","accepted","rejected"):raise ValueError("Invalid review state.")
                r["editor"].set_review_state(data["id"],data["state"])
            return self.j(w.snapshot(k))
    def route(self,post):
        try:self.handle_api(post)
        except (BrokenPipeError,ConnectionResetError):pass
        except (ValueError,OSError,KeyError,RuntimeError) as exc:
            if not isinstance(exc,(ValueError,KeyError)):LOG.exception("Local request failed")
            self.j({"error":str(exc) if isinstance(exc,(ValueError,KeyError)) else "Could not read or process this recording."},400)
    def do_GET(self):self.route(False)
    def do_POST(self):self.route(True)
    def log_message(self,*args):pass


def main():
    p=argparse.ArgumentParser(description="WhistleMamba Studio local browser")
    p.add_argument("--port",type=int,default=8765);p.add_argument("--checkpoint",default=os.environ.get("WHISTLEMAMBA_CHECKPOINT"))
    p.add_argument("--open-wav",type=Path,action="append",default=[],help="Preview this WAV at startup without running inference (repeatable).")
    args=p.parse_args()
    with tempfile.TemporaryDirectory(prefix="whistlemamba-browser-") as td:
        with ThreadingHTTPServer(("127.0.0.1",args.port),Handler) as http:
            http.workspace=Workspace(td,args.checkpoint)
            for wav in args.open_wav:
                with wav.open("rb") as src:
                    http.workspace.receive(src,os.fstat(src.fileno()).st_size,wav.name)
            print(f"WhistleMamba Studio ready at http://127.0.0.1:{http.server_port}/",flush=True)
            http.serve_forever()

if __name__=="__main__":main()

