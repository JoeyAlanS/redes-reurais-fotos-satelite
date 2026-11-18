import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from projeto3.prepare import prepare  # noqa: E402

load_dotenv(ROOT / ".env")
data_dir = Path(os.getenv("DATA_DIR", "dados/raw/NWPU-RESISC45"))
if not data_dir.is_absolute():
    data_dir = ROOT / data_dir
report = prepare(data_dir, ROOT)
print(f"Imagens inventariadas: {report['total']}; duplicatas exatas: {report['duplicatas_exatas']}; problemas: {len(report['problemas'])}")
for label, counts in report["contagens"].items():
    print(label, counts)
