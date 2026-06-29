import torch, threading
from utils import Logger

def async_save(model = None, path = ""):

    def _save():

        try:
            torch.save(model.state_dict(), path)
            Logger.logInfo(f"Model saved: {path}")

        except Exception as e:
            Logger.logInfo(f"Saving model failed: {e}")

    threading.Thread(target=_save, daemon=True).start()


def load_parameters(path = ""):

    # Load model parameters
    if (path != ""):

        try:
            parameters = torch.load(path, weights_only=True)
            Logger.logInfo(f"Parameters loaded: {path}")
            return parameters

        except Exception as e:
            Logger.logInfo(f"Loading parameters failed: {e}")
            return None
        
    else:
        return None


def save(model = None, path = ""):

    # Save model weights
    if (path != ""):
        async_save(model = model, path = path)

    else:
        Logger.logInfo("Missing filename!")
    

def save_buffer(data = None, path = ""):

    if (path != ""):

        try:
            torch.save(data, path)
            Logger.logInfo(f"Buffer saved: {path}")

        except Exception as e:
            Logger.logInfo(f"Saving buffer failed: {e}")


def load_buffer(path = ""):

    if (path != ""):

        try:
            data = torch.load(path, weights_only = True)
            Logger.logInfo(f"Buffer loaded: {path}")
            return data
            
        except Exception as e:
            Logger.logInfo(f"Loading buffer failed: {e}")
            return None
            
    else:
        return None