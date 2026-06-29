import select, sys, termios, tty, readchar
from .logger import Logger

class Keyboard:

    def __init__(self, mode="train"):
        self.mode = mode
        self.stdin_fd = sys.stdin.fileno()
        self.old_term_settings = None

    def _kbhit(self):
        return select.select([sys.stdin], [], [], 0)[0] != []

    def _read_key(self):
        try:
            self.old_term_settings = termios.tcgetattr(self.stdin_fd)
            tty.setraw(self.stdin_fd)
            key = readchar.readkey()
        finally:
            if self.old_term_settings is not None:
                termios.tcsetattr(self.stdin_fd, termios.TCSADRAIN, self.old_term_settings)
                self.old_term_settings = None
        return key

    def checkMode(self):
        if self._kbhit():
            key = self._read_key()

            if key in (readchar.key.UP, readchar.key.DOWN, readchar.key.LEFT, readchar.key.RIGHT):
                return self.mode

            if len(key) == 1:
                key = key.lower()

                if key == 'e':
                    Logger.logInfo("Eval mode")
                    self.mode = "eval"
                elif key == 't':
                    Logger.logInfo("Train mode")
                    self.mode = "train"
                elif key == 'q':
                    Logger.logInfo("Quit")
                    self.mode = "quit"
                elif key == 's':
                    Logger.logInfo("Save")
                    self.mode = "save"
                elif key == 'l':
                    Logger.logInfo("Learn")
                    self.mode = "learn"

        return self.mode

    def getActionFromInstructor(self):
        key = self._read_key()

        if key == readchar.key.UP:
            return 1
        if key == readchar.key.DOWN:
            return 0
        if key == readchar.key.LEFT:
            return 3
        if key == readchar.key.RIGHT:
            return 2

        if len(key) == 1:
            key = key.lower()

            if key == 's':
                Logger.logInfo("Save")
                self.mode = "save"
            if key == 'e':
                Logger.logInfo("Eval mode")
                self.mode = "eval"
            if key == 't':
                Logger.logInfo("Train mode")
                self.mode = "train"
            if key == 'q':
                Logger.logInfo("Quit")
                self.mode = "quit"
            if key == 'l':
                Logger.logInfo("Learn")
                self.mode = "learn"

    async def quitAsyncio(self, websocket):
        await websocket.close()
        import asyncio
        loop = asyncio.get_running_loop()
        tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
        for t in tasks:
            t.cancel()
