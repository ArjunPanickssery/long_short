import json
import random

WORD_LIST_PATH = "random_words.json"
OUTPUT_PATH = "shuffled_words.json"


def generate_shuffled_words(file_path):
    # Load words from the JSON file
    with open(file_path, "r") as file:
        original_words = json.load(file)

    # Concatenate all words into a single string
    concatenated_string = "".join(original_words)

    # Convert the string into a list of characters and shuffle it
    char_list = list(concatenated_string)
    random.shuffle(char_list)

    # Recreate the shuffled string
    shuffled_string = "".join(char_list)

    # Split the shuffled string back into words of the same lengths as the original words
    new_words = []
    index = 0
    for word in original_words:
        word_length = len(word)
        new_word = shuffled_string[index : index + word_length]
        new_words.append(new_word)
        index += word_length

    return new_words


# Example usage
if __name__ == "__main__":
    new_words = generate_shuffled_words(WORD_LIST_PATH)
    print(f"Done! Saving to {OUTPUT_PATH}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(new_words, f)
