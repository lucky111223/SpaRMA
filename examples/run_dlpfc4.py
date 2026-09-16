import sys
from pathlib import Path
from spaarma.cli import main
sys.argv.extend(["--config", str(Path(__file__).parents[1] / "configs/dlpfc4.json")]) if "--config" not in sys.argv else None
main()
