import logging
import os
from pathlib import Path

_ROOT = Path(os.path.dirname(__file__)).absolute()
logging.getLogger().setLevel(logging.INFO)