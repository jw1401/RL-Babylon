import sys, pathlib, argparse, asyncio
script_dir = pathlib.Path(__file__).parent      # Folder of this file
root = script_dir.parent                        # Root of Project (one level up)
sys.path.append(str(root / "src"))              # add src to sys-paths 
sys.path.append(str(script_dir))                # add script-folder to sys-paths 

from utils import Logger, run_envs
import trainers

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type = str)
    parser.add_argument("--trainer", type = str)
    args = parser.parse_args()
    Logger.logInfo (f"{args}")

    # Create trainer class
    trainer = trainers.make_trainer(args.trainer, config_path = args.config_path)

    run_envs(num_envs = trainer.config["NUM_ENVS"], url = trainer.config["URL"])

    try:
        asyncio.run(trainer.start_server())
        
    except Exception as e:
        Logger.logError("\nError and shutdown!\n")
        Logger.logError(e)
