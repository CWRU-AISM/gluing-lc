# Put experiments/ on sys.path so tests import the entry points and their
# helpers the way `python experiments/<family>.py` does.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
