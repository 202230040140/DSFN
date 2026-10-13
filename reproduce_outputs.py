"""Export native outputs to the current benchmark format; then evaluate them."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "MethodManagement" / "tools"))
from compatible_outputs import main

if __name__ == "__main__":
    raise SystemExit(main('DSFN'))
