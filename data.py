import os
import json
from collections import defaultdict

DATA_PATH = 'data.json'

def save_to_json(dictionary, file_name):
    # Create directory if not present
    directory = os.path.dirname(file_name)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_name, "w") as f:
        json.dump(dictionary, f)


def load_from_json(file_name) -> dict:
    with open(file_name, "r") as f:
        return json.load(f)

def load_data():
    return defaultdict(dict, load_from_json(DATA_PATH))

def save_data(data):
    save_to_json(dict(data), DATA_PATH)

