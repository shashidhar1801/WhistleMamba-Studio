"""Offline, dependency-free checks for model settings, fixture integrity and source distributions."""
import hashlib
import json
from pathlib import Path
import unittest
import os, subprocess, sys

ROOT=Path(__file__).resolve().parents[1]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
class SourceContract(unittest.TestCase):
    def test_model(self):
        r=ROOT/"app/whistlemamba/resources"
        i=json.loads((r/"csr01_identity.json").read_text()); c=json.loads((r/"csr01_inference.json").read_text())
        self.assertEqual(digest(r/"csr01_inference.json"),i["config_sha256"])
        self.assertEqual((i["overall_epoch"],i["raw_epoch"],i["state_key"]),(12,11,"model"))
        self.assertFalse(i["presence_gating"]);self.assertIsNone(i["download_url"])
        self.assertEqual(c["tracker"]["edge_threshold"],0.6)
        self.assertEqual(c["features"]["normalization"],"global_zscore")
        self.assertEqual(i["default_tracker"],"learned")
        env={**os.environ,"PYTHONPATH":str(ROOT/"app")}
        for k in ("MAMBA_DETERMINISTIC","CUBLAS_WORKSPACE_CONFIG"):
            env.pop(k,None)
        probe="import os,whistlemamba; assert os.environ['MAMBA_DETERMINISTIC']=='1' and os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'"
        subprocess.run([sys.executable,"-c",probe],env=env,check=True)
        bad=subprocess.run([sys.executable,"-c",probe],env={**env,"MAMBA_DETERMINISTIC":"0"},capture_output=True)
        self.assertNotEqual(bad.returncode,0);self.assertIn(b"fresh CSR01 process",bad.stderr)
    def test_demo(self):
        e=ROOT/"examples"
        rows=(e/"SHA256SUMS.txt").read_text().splitlines()
        self.assertEqual(len(rows),4)
        for row in rows:
            expected,name=row.split("  ",1)
            self.assertEqual(digest(e/name),expected)
    def test_no_model_payload(self):
        for p in ROOT.rglob("*"):
            if ".git" not in p.parts:
                self.assertNotIn(p.suffix.lower(),{".pt",".pth",".safetensors",".npz",".npy"})
