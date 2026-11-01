"""Set only this research process's conda DLL path, then run its supervisor."""
import os
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[1]
prefix = Path(sys.executable).parent
os.environ["PATH"] = os.pathsep.join([str(prefix), str(prefix / "Library/bin"),
                                    str(prefix / "Scripts"), os.environ.get("PATH", "")])
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "2"
script = root / "source/safe_server_queue.py"
sys.argv = [str(script), "--root", str(root / "supervisor"),
            "--queue", str(root / "configs/queue.json"), "--inputs", str(root / "inputs")]
runpy.run_path(str(script), run_name="__main__")
