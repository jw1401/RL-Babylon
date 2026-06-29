import json, os

class RLJsonLogger:

    def __init__(self, filepath="./.metrics/stats.json"):

        self.filepath = filepath
        self.data = {"episodes": [], "loss": [], "entropy": [], "score": []}

        # Ensure file exists
        if not os.path.exists(filepath):
            try:
                with open(filepath, "w") as f:
                    json.dump(self.data, f)
            except:
                print("Error: " + filepath)

    def log(self, episode, loss, entropy, score):

        self.data["episodes"].append(episode)
        self.data["loss"].append(float(loss))
        self.data["entropy"].append(float(entropy))
        self.data["score"].append(float(score))

        self.save()

    def save(self):
        
        with open(self.filepath, "w") as f:
            json.dump(self.data, f, indent=2)