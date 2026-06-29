import traceback, websockets
import numpy as np
from .bson_socket_communicator import BsonSocketCommunicator
from utils import Logger

# ============================================== #
# BabylonEnv Environment Class                   #
# Handles Enviropnement Logic                    #
# ============================================== #

class BabylonEnv:

    def __init__(self, websocket: websockets):

        self.communicator = BsonSocketCommunicator(websocket)
        
        self.state_dim  = 0         # comes from environment
        self.action_dim = 0         # discrete space: example --> up, down, left, right
        self.visual_obs = False
        self.state = None           # np.zeros(self.state_dim, dtype = np.float32)

    # ------------------------------------------------------------- #
    # Get initial data from environment like state and action space #
    # ------------------------------------------------------------- #
    async def init(self):

        try:
            await self.communicator.send({"init": True})

            # Empfange und decodiere das BSON-Paket
            decoded_bson = await self.communicator.recv()

            self.visual_obs = decoded_bson["VIS_OBS"]
            self.state_dim  = decoded_bson["state_space"]
            self.action_dim = decoded_bson["action_space"]

            Logger.logInfo("Client connected and initialized via BSON")
            Logger.logInfo(f"\033[33mVisual_Obs: {self.visual_obs}, State_Dim: {self.state_dim}, Action_Dim: {self.action_dim}\033[0m")

            return decoded_bson

        except Exception as e:
            Logger.logError(f"Unexpected error: {traceback.format_exc()}")


    # ----------------------------------- #
    # Reset the environment and wait      #
    # for incoming data with states, ... #
    # ----------------------------------- #
    async def reset(self):

        try:
            await self.communicator.send({"reset": True})

            # Empfange das Kombi-BSON-Paket
            decoded_bson = await self.communicator.recv()

            if self.visual_obs:
                _, frame = self.communicator.parse_visual_frame(decoded_bson, self.state_dim)
                self.state = frame
                
            else:
                # Daten stecken im inneren "data"-Objekt des BSONs
                inner_data = decoded_bson.get("data", {})
                self.state = np.array(inner_data["state"], dtype=np.float32)

            return self.state

        except Exception as e:
            Logger.logError(f"Unexpected error: {traceback.format_exc()}")
    

    # ---------------------------------------------- #
    # Step the environment --> send action and wait  #
    # for incoming data with states, rewards, ....   #
    # ---------------------------------------------- #
    async def step(self, action):

        try:
            await self.communicator.send({"action": action})   # int(action)

            # Empfange das Kombi-BSON-Paket
            decoded_bson = await self.communicator.recv()

            if self.visual_obs:
                inner_data, frame = self.communicator.parse_visual_frame(decoded_bson, self.state_dim)
                self.state = frame

            else:
                inner_data = decoded_bson.get("data", {})
                self.state = np.array(inner_data["state"], dtype=np.float32)
            
            reward = float(inner_data["reward"])
            terminated = bool(inner_data["terminated"]) 
            truncated =  bool(inner_data["truncated"])

            return self.state, reward, terminated, truncated

        except Exception as e:
            Logger.logError(f"Unexpected error: {traceback.format_exc()}")


    # ---------------------------------------------- #
    # Set the curriculum level.                      #
    # ---------------------------------------------- #
    async def curriculum_level(self, level):

        try:
            await self.communicator.send({"level": int(level)})

            # Empfange das Kombi-BSON-Paket
            decoded_bson = await self.communicator.recv()

            curriculum_level = decoded_bson["curriculum_level"]

            return curriculum_level

        except Exception as e:
            Logger.logError(f"Unexpected error: {traceback.format_exc()}")
