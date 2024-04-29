from models import get_model, get_all_models
from data import load_data, save_data, load_from_json
from tqdm import trange

def process_words(words, model):
    model_name = model.model_name
    data = load_data()
    for i in trange(len(words)):
        word = words[i]
        if model_name not in data[word]:
            data[word][model_name] = model.long_score(word)
        if i % 1000 == 0:
            save_data(data)
    save_data(data)    
    
def process_all_same_letter_words(model):
    words = [letter * length for letter in 'abcdefghijklmnopqrstuvwxyz' for length in range(1, 21)]
    process_words(words, model)

def process_words_from_file(model, file_name):
    words = load_from_json(f'lists/{file_name}')
    process_words(words, model)

if __name__ == '__main__':
    models = [get_model('gpt35'), get_model('gpt4')]
    
    for model in models:
        process_all_same_letter_words(model)
        process_words_from_file(model, 'random_words.json')
        process_words_from_file(model, 'shuffled_words.json')
    
    print('Done!')
