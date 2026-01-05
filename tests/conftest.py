import sys
from pathlib import Path

# adds project root to PYTHONPATH so "import src...." works
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
