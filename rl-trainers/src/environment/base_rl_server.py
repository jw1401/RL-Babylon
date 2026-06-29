import websockets, asyncio, torch
from abc import ABC, abstractmethod
from utils import Logger, ConfigLoader

class BaseRLTrainerServer(ABC):

    def __init__ (self, host = "localhost", port = 8765, config_path = ""):
        
        self.host = host
        self.port = port

        if config_path != "":
            self.config = ConfigLoader(config_path = config_path)
            self.checkCUDA(DEVICE=self.config["DEVICE"])

    @abstractmethod
    async def trainer(self, websocket: websockets):

        """
        Diese Methode ist 'blank' (abstrakt).

        Jede Klasse, die von BaseRLServer erbt, MUSS diese Methode
        implementieren und ihren eigenen Trainings-Algorithmus (PPO, DQN, etc.)
        hier implementieren.

        """
        pass

    async def start_server(self):

        async with websockets.serve(self.trainer, self.host, self.port):
            Logger.logInfo(f"RL-Trainer-Server running at ws://{self.host}':{self.port}")
            await asyncio.Event().wait()

    
    def checkCUDA(self, DEVICE = None):

        # Check CUDA Device and set actual device
        Logger.logInfo(f"CUDA available:{torch.cuda.is_available()}")
        Logger.logInfo(f"MPS available : {torch.backends.mps.is_available()}")

        if torch.cuda.is_available():
            Logger.logInfo(f"Available GPU is {torch.cuda.get_device_name(0)}")

        ActualDEVICE = torch.device(DEVICE) 

        Logger.logInfo (f"Active device is {str(ActualDEVICE).upper()}") 

        