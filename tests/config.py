import sys
import os
# Add the absolute path of the module directory (irt) to the system path
MODULE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "irtorch"))
sys.path.append(MODULE_DIR)
SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "other-scripts"))
sys.path.append(SCRIPT_DIR)
