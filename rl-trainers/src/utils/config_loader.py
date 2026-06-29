import json
from pathlib import Path

class ConfigLoader:

    _instance = None

    def __new__(cls, config_path=""):

        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance

    def _load_config(self, config_path):

        BASE_DIR = Path(__file__).resolve().parent      # is C:/..../src/Utils
        path = BASE_DIR.parent.parent / config_path     # now is in RL_Trainers folder
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            self.params = json.load(f)

    def __getitem__(self, key):

        return self.params[key]

    def __repr__(self):
        
        return f"PPOConfig({self.params})"
