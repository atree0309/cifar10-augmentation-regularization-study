"""Create a project-only reproduction environment using the caller's Python 3.15."""
import os
import subprocess
import sys
import sysconfig
import venv
from pathlib import Path

root=Path(__file__).resolve().parents[2]
if sys.version_info[:2]!=(3,15):
    raise SystemExit('Run with Python 3.15, for example: py -3.15 outputs/cifar10/setup_environment.py')
target=root/'work'/'venv315'
if target.exists():
    raise SystemExit('Environment already exists; inspect and reuse it rather than overwriting it.')
venv.EnvBuilder(with_pip=True).create(target)
python=target/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
subprocess.run([str(python),'-m','pip','install','torch==2.14.1','torchvision==0.29.1',
    '--index-url','https://download.pytorch.org/whl/cpu'],check=True)
lock=Path(__file__).with_name('requirements-lock.txt')
subprocess.run([str(python),'-m','pip','install','-r',str(lock),
    '--extra-index-url','https://download.pytorch.org/whl/cpu'],check=True)
if os.name=='nt':
    # This changes only this new environment, not Windows or the system Python.
    site=subprocess.check_output([str(python),'-c','import sysconfig; print(sysconfig.get_paths()["purelib"])'],text=True).strip()
    hook='import os, ctypes; _project_msvcp_runtime = ctypes.WinDLL(os.path.join(os.environ.get("SystemRoot", r"C:\\Windows"), "System32", "msvcp140.dll"))\n'
    (Path(site)/'project_runtime_fix.pth').write_text(hook,encoding='ascii')
subprocess.run([str(python),'-c','import torch,torchvision; print(torch.__version__,torchvision.__version__)'],check=True)
print('Environment ready:',python)
