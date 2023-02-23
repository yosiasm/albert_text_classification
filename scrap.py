import requests
from tqdm import tqdm
import time
import json

tags = [
    "api",
    "database",
    "sql",
    "nosql",
    "frontend",
    "backend",
    "network",
    "docker",
    "pipeline",
    "devops",
    "sre",
    "model",
    "deployment",
    "mobile",
    "cloud",
    "server",
    "ml",
]
for tag in tags:
    posts = []
    loop = 25
    for page in tqdm(range(1, loop + 1), desc=tag):
        trial = True
        # retry when error
        while trial:
            try:
                url = f"https://api.stackexchange.com/2.3/search/advanced?order=desc&page={page}&pagesize=100&sort=activity&accepted=True&tagged={tag}&title=error&site=stackoverflow"
                data = requests.get(url).json()
                posts.extend(data["items"])
                trial = False
            except Exception as e:
                print(e)
                time.sleep(20)
        time.sleep(7)
    with open(f"data_error/{tag}.json", "w") as w:
        json.dump(posts, w)
