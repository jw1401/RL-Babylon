import torch, numpy as np

class StateStacker:
    
    def __init__(self, obs_shape, num_stack = 4, device = "cpu"):

        self.num_stack = num_stack
        self.device = device
        
        # Buffer mit Nullen initialisieren
        self.stack = np.zeros(obs_shape, dtype = np.float32)    

    def reset(self, initial_obs):

        # Beim Episode-Start Buffer mit Startzustand füllen.

        for i in range(self.num_stack):
            self.stack[i] = initial_obs

        return self._get_stacked()

    def push(self, new_obs):

        # Neuen Zustand hinzufügen und ältesten verwerfen.

        self.stack = np.roll(self.stack, shift=-1, axis=0)
        self.stack[-1] = new_obs
        return self._get_stacked()

    def _get_stacked(self):

        # Stack als Tensor zurückgeben.

        stacked = torch.tensor(self.stack, device=self.device)

        # Flatten oder Reshape – je nach Netzarchitektur
        return stacked 
