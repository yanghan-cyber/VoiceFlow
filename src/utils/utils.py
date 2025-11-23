import getpass
import os
from .log_utils import get_logger
def set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


logger = get_logger("default", "logs/default.log", level="DEBUG")

