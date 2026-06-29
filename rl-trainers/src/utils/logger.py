from .rl_json_logger import RLJsonLogger 

class Logger:

    def __init__(self, print_interval = 200):

        self.print_interval = print_interval
        self.jsonLogger = RLJsonLogger()

    @staticmethod
    def log(level=" Zero ",message=""):
        print(f"[{level}]: {message}")

    @staticmethod
    def logInfo(message):
        Logger.log(" Info ", message)

    @staticmethod
    def logError(error):
        Logger.log(" Err  ", f"\033[31m{error}\033[0m")

    # Log stats every print interval
    #
    def logStats(self, episode = 0, score= 0, loss = 0, policy_loss = 0, value_loss = 0 ,entropy = 0):

        if episode % self.print_interval == 0 and episode != 0:
            
            self.jsonLogger.log(episode, loss, entropy, score/self.print_interval)
            message = f"\033[1;32mEpisode: {episode}, Avg-Score: {score/self.print_interval:.2f}, Loss: {loss:.2f}, Policy: {policy_loss:.2f}, Value: {value_loss:.2f}, Entropy: {entropy:.2f} \033[0m"
            Logger.log(" Stat ", message)
            return True
        
        return False
