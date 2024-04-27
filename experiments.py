from models import get_model, get_all_models
from data import load_data, save_data
from tqdm import tqdm

def process_words(words, model):
    model_name = model.model_name
    data = load_data()
    for word in tqdm(words):
        if model_name not in data[word]:
            data[word][model_name] = model.long_score(word)

    save_data(data)    
    
if __name__ == '__main__':
    gpt35 = get_model("gpt35")
    process_words(['a', 'aa', 'aaa'], gpt35)
    pass