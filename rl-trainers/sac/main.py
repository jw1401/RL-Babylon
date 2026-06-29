import sys, pathlib, argparse, asyncio
script_dir = pathlib.Path(__file__).parent      # Folder of this file
root = script_dir.parent                        # Root of Project (one level up)
sys.path.append(str(root / "src"))              # add src to sys-paths 
sys.path.append(str(script_dir))                # add script-folder to sys-paths 

from utils import Logger, run_envs
from trainer import RLTrainerServer

if __name__ == "__main__":

    # Create trainer class
    trainer = RLTrainerServer()
    run_envs(num_envs = 1, url = "http://localhost:5500/babylon-environments/Cube-Ball/Continous-SAC-Demo/Environment.html")

    try:
        asyncio.run(trainer.start_server())
        
    except Exception as e:
        Logger.logError(f"Error: {e}")
