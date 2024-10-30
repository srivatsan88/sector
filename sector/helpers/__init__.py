
import importlib
import os

__all__ = []

current_dir = os.path.dirname(__file__)

# Loop over each file in the directory
for filename in os.listdir(current_dir):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = filename[:-3]  # Strip .py extension
        module = importlib.import_module(f"sector.helpers.{module_name}")

        # Add all public attributes to the current namespace
        for attribute in dir(module):
            if not attribute.startswith("_"):
                globals()[attribute] = getattr(module, attribute)
                __all__.append(attribute)