import traceback, websockets, bson
import numpy as np
from PIL import Image
from utils import Logger

# ============================================== #
# WebSocket Communicator Class                   #
# Handles communication with Babylon environment #
# ============================================== #

class BsonSocketCommunicator:

    def __init__(self, websocket: websockets):

        self.ws = websocket
        self.debug = False
        
    # ------------------------------------ #
    # Send data encoded in BSON format     #
    # ------------------------------------ #
    async def send(self, msg):
        
        try:
            # Objekt direkt in binäres BSON serialisieren und senden
            bson_packet = bson.encode(msg)
            await self.ws.send(bson_packet)

        except Exception as e:
            Logger.logError(f"Unexpected error: {traceback.format_exc()}")
        

    # --------------------------------------- #
    # Receive data and convert from BSON      #
    # --------------------------------------- #
    async def recv(self):

        try:
            # Empfängt das rohe binäre BSON-Paket vom JS-Client
            data = await self.ws.recv()
            
            if isinstance(data, (bytes, bytearray)):
                # Direkt in ein Python-Dictionary decodieren
                return bson.decode(data)
            else:
                Logger.logError(f"Expected binary bytes from WebSocket, got {type(data)}")
                return None

        except Exception as e:
            Logger.logError(f"Unexpected error: {traceback.format_exc()}")
            return None


    # --------------------------------------- #
    # Parse BSON document and convert image   #
    # --------------------------------------- #
    def parse_visual_frame(self, decoded_bson, state_dim):

        # Extrahiere das verschachtelte RL-Datenobjekt
        rl_data = decoded_bson.get("data", {})
        
        # Extrahiere die rohen Bild-Bytes aus dem BSON-Feld
        pixels = decoded_bson.get("frame")

        if pixels is None:
            Logger.logError("No pixels found in BSON packet")
            return rl_data, "No pixels"
        
        # Da pixels ein nativer 'bytes'-Typ ist, können wir ihn direkt in numpy einlesen
        frame = np.frombuffer(pixels, np.uint8).reshape((state_dim[0], state_dim[1], 4))  # raw frame from babylon
        frame = frame[:,:,:3]       # without alpha channel
        frame = np.flipud(frame)    # frame flipped 

        gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])  # convert to grayscale
        gray = gray.astype(np.uint8)

        if self.debug:

            Image.fromarray(frame, 'RGB').show()
            Image.fromarray(gray, 'L').show()
            input("Press Enter to continue...")

        return rl_data, gray / 255.0   # /255 for normalization


    
