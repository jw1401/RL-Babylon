import msvcrt
from .logger import Logger 

class Keyboard:

    def __init__(self, mode = "train"):

        self.mode = mode
        
    # Check mode in RL cycle
    def checkMode(self):

        # check if a key is pressed    
        if msvcrt.kbhit():           
            
            key = msvcrt.getch()     
            key = key.decode('utf-8').lower()

            if key == 'e':
                Logger.logInfo("Eval mode")
                self.mode="eval"
                
            elif key == 't':
                Logger.logInfo("Train mode")
                self.mode="train"
                
            elif key == 'q':
                Logger.logInfo("Quit")
                self.mode="quit"

            elif key == 's':
                Logger.logInfo("Save")
                self.mode ="save"

            elif key == 'l':
                Logger.logInfo("Learn")
                self.mode ="learn"
                
        return self.mode
    
    
    # Get actions from keyboard
    def getActionFromInstructor(self):

        key = msvcrt.getch()

        if key in (b'\xe0', b'\x00'):           # Arrow keys start with \xe0 or \x00

            c2 = msvcrt.getch()                 # zweites Byte, blockiert

            if c2 == b'H':                      # Up
                return 1
            elif c2 == b'P':                    # Down
                return 0
            elif c2 == b'K':                    # Left
                return 3
            elif c2 == b'M':                    # Right
                return 2

        else:

            key = key.decode('utf-8').lower()

            if key == 's':
                Logger.logInfo("Save")
                self.mode ="save"

            if key == 'e':
                Logger.logInfo("Eval mode")
                self.mode ="eval"   

            if key == 't':
                Logger.logInfo("Train mode")
                self.mode="train" 
            
            if key == 'q':
                Logger.logInfo("Quit")
                self.mode="quit"
            
            if key == 'l':
                Logger.logInfo("Learn")
                self.mode ="learn"
