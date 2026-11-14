import subprocess
import sys
from pathlib import Path
import pytest

@pytest.fixture(scope="session")
def synthetic_wav(tmp_path_factory):
    directory = tmp_path_factory.mktemp("generated-audio")
    subprocess.run([sys.executable, str(Path(__file__).with_name("generate_fixture.py")), str(directory)], check=True)
    return directory / "synthetic_whistles_4s.wav"

@pytest.fixture(autouse=True)
def sample_for_file_tests(request, monkeypatch):
    for module_name, variable in (("test_deployment", "demo"), ("test_recording_intake", "DEMO")):
        if request.module.__name__.split(".")[-1] == module_name:
            monkeypatch.setattr(request.module, variable, request.getfixturevalue("synthetic_wav"))
