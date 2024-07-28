import os
from abc import ABC, abstractmethod
from collections import Counter
from math import exp

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from openai import OpenAI
from random_word import RandomWords
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


class ModelWrapper(ABC):
    def __init__(self, config=None):
        self.config = config

    @abstractmethod
    def format_prompt(self, user_prompt, system_prompt):
        pass

    @abstractmethod
    def generate(self, formatted_prompt, device="cuda", max_length=100):
        pass

    @abstractmethod
    def get_probs(self, input_text, tokens):
        pass

    @abstractmethod
    def long_score(self, word):
        pass


class LlamaModel(ModelWrapper):
    def __init__(self, config=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: ", self.device)

        self.model_name = config["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=HF_TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", token=HF_TOKEN
        )
        # self.model.to('cuda')

    def format_prompt(
        self,
        user_prompt,
        system_prompt="You are a helpful assistant.",
        words_in_mouth="",
    ):
        match self.model_name:
            case "meta-llama/Meta-Llama-3-8B-Instruct":
                return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{words_in_mouth}"""

            case "meta-llama/Llama-2-7b-chat-hf":
                words_in_mouth = " " + words_in_mouth
                return f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>
{user_prompt} [/INST]{words_in_mouth}"""

    def generate(self, formatted_prompt, device="cuda", max_length=100):
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(
            device
        )
        output = self.model.generate(input_ids, max_new_tokens=max_length)
        return (
            self.tokenizer.decode(output[0].to("cpu"), skip_special_tokens=True)
            .split("[/INST]")[-1]
            .strip()
        )

    def get_probs(self, formatted_prompt, tokens):
        # Prepare the input
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(
            self.model.device
        )

        # Perform a forward pass
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            outputs = self.model(input_ids)

        # Extract logits
        logits = outputs.logits

        # Select the logits for the first token position after the input
        first_position_logits = logits[0, len(input_ids[0]) - 1, :]

        # Apply softmax to get probabilities
        probs = F.softmax(first_position_logits, dim=-1)

        res = {}
        for token in tokens:
            res[token] = probs[
                self.tokenizer.encode(token, add_special_tokens=False)[-1]
            ].item()

        return res

    def get_probs_batched(self, formatted_prompts, tokens):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = torch.tensor(
            self.tokenizer.batch_encode_plus(
                formatted_prompts, padding="longest"
            ).input_ids
        ).to(
            self.model.device
        )  # pad so we can turn seqs into batch (tensor needs all inner dims to be same)
        extra_padding = (
            (torch.ones((len(formatted_prompts), 1)) * self.tokenizer.eos_token_id)
            .int()
            .to(self.model.device)
        )
        # Concatenate inputs and extra padding
        # to ensure there are _always_ EOS tokens at end
        inputs = torch.cat((inputs, extra_padding), dim=-1)
        output = self.model(inputs).logits

        mask = (
            inputs == self.tokenizer.eos_token_id
        ).int()  # sub eos for a diff token id, if you want to read some diff position
        first_occurrences = torch.argmax(mask, dim=1)
        last_token_idx = first_occurrences - 1

        last_token_logits = output[
            torch.arange(output.size(0)), last_token_idx, :
        ]  # take every index for the batch dim, only the last token for the ctx dim, and keep the vocab size dim
        last_token_probs = torch.softmax(
            last_token_logits, dim=-1
        )  # softmax over dict dim to get prob distr over all tokens in dict

        first_token_id = self.tokenizer.encode(tokens[0])[
            -1
        ]  # discard start of seq token
        second_token_id = self.tokenizer.encode(tokens[1])[-1]

        p_first_token = last_token_probs[:, first_token_id]
        p_second_token = last_token_probs[:, second_token_id]

        return [
            dict([(tokens[0], p_first_token[i]), (tokens[1], p_first_token[i])])
            for i in range(len(p_first_token))
        ]

    def long_score(self, word):
        OUTPUTS = [" long", " short"]
        scores = self.get_probs(
            self.format_prompt(
                user_prompt=f'Is this word long or short? Only say "long" or "short". The word is: {word}.',
                words_in_mouth="That word is...",
            ),
            OUTPUTS,
        )
        return scores[OUTPUTS[0]] / sum(scores.values())

    def long_score_batched(self, words):
        OUTPUTS = [" long", " short"]
        formatted_prompts = [
            self.format_prompt(
                user_prompt=f'Is this word long or short? Only say "long" or "short". The word is: {word}.',
                words_in_mouth="That word is...",
            )
            for word in words
        ]
        scores = self.get_probs_batched(formatted_prompts, OUTPUTS)
        return [score[OUTPUTS[0]] / sum(score.values()) for score in scores]


class GPTModel(ModelWrapper):
    def __init__(self, config=None):
        self.model_name = config["model_name"]
        self.openai_client = OpenAI()

    def format_prompt(self, user_prompt, system_prompt):
        pass

    def generate(self, formatted_prompt, device="cuda", max_length=100):
        history = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": formatted_prompt,
            },
        ]

        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=history,
            max_tokens=max_length,
            temperature=0,
        )

        return response.choices[0].message.content

    def get_probs(self, formatted_prompt, tokens):
        def get_logits(prompt):
            history = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]

            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=history,
                max_tokens=1,
                temperature=0,
                logprobs=True,
                top_logprobs=5,
            )
            return response.choices[0].logprobs.content[0].top_logprobs

        logits = get_logits(formatted_prompt)
        return {item.token: exp(item.logprob) for item in logits}

    def long_score(self, word):
        OUTPUTS = ["long", "short"]
        scores = self.get_probs(
            f'Is this word long or short? Only say "long" or "short". The word is: {word}.',
            OUTPUTS,
        )
        if OUTPUTS[0] in scores and OUTPUTS[1] not in scores:
            return 1
        if OUTPUTS[1] in scores and OUTPUTS[0] not in scores:
            return 0
        if OUTPUTS[0] in scores and OUTPUTS[1] in scores:
            return scores[OUTPUTS[0]] / (scores[OUTPUTS[0]] + scores[OUTPUTS[1]])
        return 0.5


def get_model(model_name):
    match model_name:
        case "gpt35":
            return GPTModel({"model_name": "gpt-3.5-turbo-0125"})
        case "gpt4":
            return GPTModel({"model_name": "gpt-4"})
        case "llama2":
            return LlamaModel({"model_name": "meta-llama/Llama-2-7b-chat-hf"})
        case "llama3":
            return LlamaModel({"model_name": "meta-llama/Meta-Llama-3-8B-Instruct"})


def get_all_models():
    gpt35 = GPTModel({"model_name": "gpt-3.5-turbo-0125"})
    gpt4 = GPTModel({"model_name": "gpt-4"})
    llama2 = LlamaModel({"model_name": "meta-llama/Llama-2-7b-chat-hf"})
    llama3 = LlamaModel({"model_name": "meta-llama/Meta-Llama-3-8B-Instruct"})

    return [gpt35, gpt4, llama2, llama3]
