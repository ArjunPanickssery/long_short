from random_word import RandomWords
from tqdm import tqdm
import json

NUM_WORDS = 10000
FILE_NAME = "random_words.json"

if __name__ == "__main__":
    r = RandomWords()

    words = []
    for _ in tqdm(range(NUM_WORDS)):
        words.append(r.get_random_word())

    with open(FILE_NAME, "w") as f:
        json.dump(words, f)

    print(f"Done! Saved to {FILE_NAME}")
