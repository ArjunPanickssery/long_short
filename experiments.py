from models import get_all_models

gpt35, gpt4, llama2, llama3 = get_all_models()

words2 = [
    "a",
    "the",
    "this",
    "bench",
    "hallowed",
    "screenshotted",
    "Californication",
    "abracadabra",
    "sdjfiagljagdlfug",
]
llama3.long_score_batched(words2)
