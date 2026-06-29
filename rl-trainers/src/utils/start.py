import sys, subprocess

def open_chrome_incognito(URL =""):

    try:

        if sys.platform == "win32":
            subprocess.Popen(["C:/Program Files/Google/Chrome/Application/chrome.exe", "--incognito", "--new-window", "--window-size=500,500", URL])

        elif sys.platform == "darwin":
            # On macOS, directly launch Chrome binary for reliable window opening
            subprocess.Popen(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome","--incognito", "--new-window", "--window-size=500,500", URL],stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

        else:
            subprocess.Popen(["google-chrome", "--incognito", "--new-window", "--window-size=500,500", URL])

    except Exception as e:

        print(f"Warning: Could not open Chrome directly: {e}")
        

def run_envs(num_envs=1, url=""):

    for n in range(num_envs): 
        open_chrome_incognito(URL=url)
    





