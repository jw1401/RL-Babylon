import sys, pathlib, asyncio

script_dir = pathlib.Path(__file__).parent      # Folder of this file
root = script_dir.parent                        # Root of Project (one level up)
sys.path.append(str(root / "src"))              # add src to sys-paths 
sys.path.append(str(script_dir))                # add script-folder to sys-paths 

env_case = 1                                    # 1 --> single, 2 --> mult, 3 --> continous

from utils import run_envs
 
if __name__ == "__main__":

    if env_case == 1:
        from single_env_test import RLTrainerServer     
        trainer_server = RLTrainerServer()
        run_envs(num_envs = 1, url = "http://localhost:5500/babylon-environments/Cube-Ball/Vector-Obs/Environment.html")

    elif env_case ==2:
        from mult_env_test import RLTrainerServer
        trainer_server = RLTrainerServer(num_envs = 3)
        run_envs(num_envs = 3, url = "http://localhost:5500/babylon-environments/Cube-Ball/Vector-Obs/Environment.html")

    elif env_case ==3:
        from continous_env_test import RLTrainerServer
        trainer_server = RLTrainerServer()
        run_envs(num_envs = 1, url = "http://localhost:5500/babylon-environments/Cube-Ball/Continous-SAC-Demo/Environment.html")

    try:
        asyncio.run(trainer_server.start_server())
        
    except Exception as e:
      print(f"Error: {e}")
