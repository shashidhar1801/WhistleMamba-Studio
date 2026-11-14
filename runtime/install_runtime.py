"""Run in Ubuntu-24.04 with /usr/bin/python3.12. No sudo, model or audio access."""
import argparse, hashlib, json, os, platform, subprocess, sys, tempfile
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheelhouse',type=Path,help='Optional directory of matching wheels for a fully offline install')
    args=parser.parse_args()
    if sys.version_info[:2]!=(3,12) or sys.platform!='linux' or platform.machine()!='x86_64':
        parser.error('Use Ubuntu 24.04 in WSL 2, x86-64, Python 3.12 (not Windows Python).')
    wheels=args.wheelhouse.resolve(strict=True) if args.wheelhouse else None
    manifest=Path(__file__).with_name('runtime.lock.json').read_bytes()
    deps=json.loads(manifest); tag=hashlib.sha256(manifest).hexdigest()
    root=Path.home()/'.local/share/whistlemamba-studio/csr01-e12'
    dst=root/'venv'; py=dst/'bin/python'; marker=dst/'studio-runtime.sha256'
    env={**os.environ,'PYTHONNOUSERSITE':'1','PIP_CONFIG_FILE':os.devnull}
    for key in ('PYTHONPATH','PYTHONHOME','PIP_EXTRA_INDEX_URL','PIP_INDEX_URL','PIP_FIND_LINKS','PIP_TRUSTED_HOST'):
        env.pop(key,None)
    def run(cmd): subprocess.run(cmd,env=env,check=True)
    def check():
        run([py,'-I','-m','pip','check'])
        run([py,'-I','-c',"import torch; assert torch.__version__=='2.9.1+cu128' and torch.cuda.is_available(), 'WSL CUDA GPU not available'; from mamba_ssm import Mamba2; import causal_conv1d_cuda, selective_scan_cuda; print(torch.cuda.get_device_name(0))"])
    if dst.exists():
        if not marker.is_file() or marker.read_text().strip()!=tag:
            parser.error(f'Existing environment not owned by this completed setup: {dst}. It was not changed. Choose a verified existing interpreter in Configure Detector, or move this environment aside yourself before retrying.')
        check();print('Already installed and checked:',py);return
    root.mkdir(parents=True,exist_ok=True)
    # Build at its permanent virtualenv path; venv launchers embed this path.
    dst.mkdir()
    print('Creating isolated detector runtime:',dst,flush=True)
    run([sys.executable,'-I','-m','venv',str(dst)])
    with tempfile.TemporaryDirectory() as td:
        req=Path(td)/'requirements.txt'
        req.write_text(''.join((r['name']+' @ '+r['url'] if r.get('url') and not wheels else r['name']+'=='+r['version'])+' --hash=sha256:'+r['sha256']+'\n' for r in deps))
        cmd=[py,'-I','-m','pip','--isolated','--disable-pip-version-check','install','--no-input','--only-binary=:all:','--require-hashes','--no-deps']
        cmd+=(['--no-index','--find-links',str(wheels)] if wheels else ['--index-url','https://pypi.org/simple'])
        run(cmd+['-r',req])
    check();marker.write_text(tag+'\n');print('Ready:',py)

if __name__=='__main__':main()
