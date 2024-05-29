"""
Generate samples with GPT-2 and filter out those that are likely to be
memorized samples from the training set.
"""

import logging
import json
logging.basicConfig(level='ERROR')
import pytorch_lightning as pl
from models.Neo_Model import Neo
from models.Neo_Model_valid import NeoValid
from models.Neo_Model_suffix_tree import NeoST
from models.Neo_Model_DP import NeoDP
import argparse
import numpy as np
from pprint import pprint
import sys
import torch
import zlib
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculatePerplexity(sentence, model, tokenizer):
    """
    exp(loss)
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return torch.exp(loss)

def print_best(metric, samples, name1, scores1, name2=None, scores2=None, n=10):
    """
    print the `n` best samples according to the given `metric`
    """
    idxs = np.argsort(metric)[::-1][:n]

    for i, idx in enumerate(idxs):
        if scores2 is not None:
            print(f"{i+1}: {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}")
        else:
            print(f"{i+1}: {name1}={scores1[idx]:.3f}, , score={metric[idx]:.3f}")

        print()
        #for line in samples[idx].split("\n"):
        #    print(f"\t {line.rstrip()}")
        pprint(samples[idx])
        print()
        print()
        

def parse_commoncrawl(wet_file):
    """
    Quick and ugly parsing of a WET file.
    Tested for the May 2021 crawl.
    """
    with open(wet_file) as f:
        lines = f.readlines() 
    
    start_idxs = [i for i in range(len(lines)) if "WARC/1.0" in lines[i]]
    
    all_eng = ""

    count_eng = 0
    for i in range(len(start_idxs)-1):
        start = start_idxs[i]
        end = start_idxs[i+1]
        if "WARC-Identified-Content-Language: eng" in lines[start+7]:
            count_eng += 1
            for j in range(start+10, end):
                all_eng += lines[j]

    return all_eng

def read_csv_to_string(filename):
    with open(filename, 'r', newline='') as csvfile:
        csv_string = csvfile.read()
    return csv_string


def includes_part(string1, string2):
    for i in range (max(len(string1)-40,1), (max(len(string1)-20,1))):
        if string2.find(string1[i:]) != -1:
            print(string1[i:])
            return True
    return False


def main():
    print(f"using device: {device}")

    # if args.internet_sampling:
    #     print("Loading common crawl...")
    #     cc = parse_commoncrawl(args.wet_file)

    if args.internet_sampling:
        print("Loading 5 csv files...")
        cc = read_csv_to_string(args.wet_file)

    # print(cc.type)
    # print(cc[:1000])
    # print(cc[10])
    # print(cc[20])

    # number of tokens to generate
    seq_len = 256

    # sample from the top_k tokens output by the model
    top_k = 40


    print("Loading NEO...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token
    # model_path = "data2/bhkim/unlearning_ckpt_13"
    model_path = "../../data2/bhkim/unlearning_ckpt_13"

    # model1 =  AutoModelForCausalLM.from_pretrained(model_path).to(device)

    
    
    # model1 =  AutoModelForCausalLM.from_pretrained(
    #             "EleutherAI/gpt-neo-125M",
    #             resid_dropout=0,
    #             embed_dropout=0,
    #             attention_dropout=0,
    #             pad_token_id=tokenizer.eos_token_id).to(device)
    config_path = "LM_Memorization/configs/example.json"
    with open(config_path) as config_file:
        config = json.load(config_file)
    config = argparse.Namespace(**config)

    # Init configs that are not given
    if 'seed' not in config:
        seed = 42
    if 'privacy_method' not in config:
        config.privacy_method = None
    if 'train_sets' not in config:
        config.train_sets = ""
    if 'valid_sets' not in config:
        config.valid_sets = []
    if 'valid_subset_path' not in config:
        config.valid_subset_path = None
    if 'valid_type_path' not in config:
        config.valid_type_path = None
    if 'learning_rate' not in config:
        config.learning_rate = 5e-5
    if 'negative_loss' not in config:
        config.negative_loss = True
    if 'gradient_accumulation_steps' not in config:
        config.gradient_accumulation_steps = 1
    if 'num_train_epochs' not in config:
        config.num_train_epochs = 0
    if 'num_workers' not in config:
        config.num_workers = 0
    if 'wandb_log' not in config:
        config.wandb_log = False
    if 'strategy' not in config:
        config.strategy = None
    if 'fp16' not in config:
        config.fp16 = False
    if 'check_validation_only' not in config:
        config.check_validation_only = False
    if 'check_val_every_n_epoch' not in config:
        config.check_val_every_n_epoch = 1
    if 'tokenizer' not in config:
        config.tokenizer_name_or_path = config.model_name_or_path
    if 'target_length' not in config:
        config.target_length = None
    if 'el_n' not in config:
        config.el_n = [10]
    if 'el_threshold' not in config:
        config.el_threshold = 0
    if 'ma_threshold' not in config:
        config.ma_threshold = 0
    if 'min_train_epochs' not in config:
        config.min_train_epochs = 0
    if 'do_init_eval' not in config:
        config.do_init_eval = True if config.mode == 'unlearn' else False

    pl.seed_everything(seed, workers=True)

    # model = Neo(config)
    model1 = NeoValid.load_from_checkpoint(model_path)
    model1.eval()

    # print("Loading GPT2...")
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # tokenizer.padding_side = "left" 
    # tokenizer.pad_token = tokenizer.eos_token

    # model1 = GPT2LMHeadModel.from_pretrained('gpt2-xl', return_dict=True).to(device)
    # model1.config.pad_token_id = model1.config.eos_token_id
    # model2 = GPT2LMHeadModel.from_pretrained('gpt2', return_dict=True).to(device)
    # model1.eval()
    # model2.eval()

    samples = []
    scores = {"XL": [], "S": [], "Lower": [], "zlib": []}
    cnt = 0
    num_batches = int(np.ceil(args.N / args.batch_size))
    with tqdm(total=args.N) as pbar:
        for i in range(num_batches):
            # encode the prompts
            if args.internet_sampling:
                # pick a random 10-token prompt in common crawl 

                input_len = 10
                input_ids = []
                attention_mask = []

                while len(input_ids) < args.batch_size:
                    # take some random words in common crawl
                    r = np.random.randint(0, len(cc))
                    prompt = " ".join(cc[r:r+100].split(" ")[1:-1])

                    # make sure we get the same number of tokens for each prompt to enable batching
                    inputs = tokenizer(prompt, return_tensors="pt", max_length=input_len, truncation=True)
                    if len(inputs['input_ids'][0]) == input_len:
                        input_ids.append(inputs['input_ids'][0])
                        attention_mask.append(inputs['attention_mask'][0])

                inputs = {'input_ids': torch.stack(input_ids), 
                          'attention_mask': torch.stack(attention_mask)}

                # the actual truncated prompts
                prompts = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
            else:
                prompts = ["<|endoftext|>"] * args.batch_size
                input_len = 1
                inputs = tokenizer(prompts, return_tensors="pt", padding=True)

            # batch generation
            output_sequences = model1.generate(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                max_length=input_len + seq_len,
                do_sample=True, 
                top_k=top_k, 
                top_p=1.0
            )

            texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    
            for text in texts:
                if includes_part(text, cc):
                    print(text)
                    cnt+=1
                # perplexity of GPT2-XL and GPT2-S
                p1 = calculatePerplexity(text, model1, tokenizer)
                # p2 = calculatePerplexity(text, model2, tokenizer)

                # perplexity on lower-case sample
                p_lower = calculatePerplexity(text.lower(), model1, tokenizer)

                # Zlib "entropy" of sample
                zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))

                samples.append(text)
                # scores["XL"].append(p1)
                # scores["S"].append(p2)
                # scores["Lower"].append(p_lower)
                # scores["zlib"].append(zlib_entropy)
                # print(zlib_entropy)
                scores["XL"].append(p1.item())
                # scores["S"].append(p2.item())
                scores["Lower"].append(p_lower.item())
                scores["zlib"].append(zlib_entropy)


            pbar.update(args.batch_size)
        
    # print(scores["XL"])
    print("total inclusion")
    print(cnt)
    scores["XL"] = np.asarray(scores["XL"])
    # scores["S"] = np.asarray(scores["S"])
    scores["Lower"] = np.asarray(scores["Lower"])
    scores["zlib"] = np.asarray(scores["zlib"])

    # scores["XL"] = scores["XL"].cpu().numpy()
    # scores["S"] = scores["S"].cpu().numpy()
    # scores["Lower"] = scores["Lower"].cpu().numpy()
    # scores["zlib"] = scores["zlib"].cpu().numpy()

    # Sort by perplexity
    # metric = -np.log(scores["XL"])
    # print(f"======== top sample by XL perplexity: ========")
    # print_best(metric, samples, "PPL", scores["XL"])
    # print()
    # print()

    # # Sort by ratio of log perplexities of S and XL models
    # metric = np.log(scores["S"]) / np.log(scores["XL"])
    # print(f"======== top sample by ratio of S and XL perplexities: ========")
    # print_best(metric, samples, "PPL-XL", scores["XL"], "PPL-S", scores["S"])
    # print()
    # print()

    # Sort by ratio of log perplexities of lower-case and normal-case perplexities 
    # metric = np.log(scores["Lower"]) / np.log(scores["XL"])
    # print(f"======== top sample by ratio of lower-case and normal-case perplexities: ========")
    # print_best(metric, samples, "PPL-XL", scores["XL"], "PPL-XL-Lower", scores["Lower"])
    # print()
    # print()

    # Sort by ratio of Zlib entropy and XL perplexity
    # metric = scores["zlib"] / np.log(scores["XL"])
    # print(f"======== top sample by ratio of Zlib entropy and XL perplexity: ========")
    # print_best(metric, samples, "PPL-XL", scores["XL"], "Zlib", scores["zlib"])

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=1000, help="Number of samples to generate")
    parser.add_argument('--batch-size', type=int, default=10, help="Batch size for generation")
    parser.add_argument('--internet-sampling', action='store_true', help="condition the generation using commoncrawl")
    parser.add_argument('--wet-file', type=str, default=None, help="path to a commoncrawl WET file")
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
