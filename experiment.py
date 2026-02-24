import sys
import os
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


def read_lines(path):
    """Read a file and return a list of non-empty stripped lines."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ── Parse arguments & read input files ────────────────────────────────────────
import argparse

parser = argparse.ArgumentParser(description="Run generation + evaluation experiment.")
parser.add_argument("--llm", type=int, required=True,
                    help="0-based index of the LLM to use (from 'llms' file).")
parser.add_argument("--mode", choices=["pro-pro", "pro-perc"], required=True,
                    help="'pro-pro': eval template from pro_temp; 'pro-perc': from perc_temp.")
parser.add_argument("--pro-template", type=int, required=True,
                    help="0-based index of the production template (from 'pro_temp').")
parser.add_argument("--eval-template", type=int, required=True,
                    help="0-based index of the evaluation template.")
parser.add_argument("--max-tokens", type=int, default=100,
                    help="Maximum number of tokens to generate (default: 100).")
parser.add_argument("--output", "-o", type=str, default=None,
                    help="Output filename. If omitted, auto-derived from run parameters.")
parser.add_argument("--max-outputs", type=int, default=None,
                    help="Stop after generating this many outputs.")
args = parser.parse_args()

llms        = read_lines("llms")
pro_temps   = [t.replace("\\n", "\n") for t in read_lines("pro_temp")]
perc_temps  = [t.replace("\\n", "\n") for t in read_lines("perc_temp")]
vars_list   = read_lines("vars")

def _check_index(value, lst, name):
    if value < 0 or value >= len(lst):
        print(f"Error: --{name} {value} is out of range (0–{len(lst)-1}).", file=sys.stderr)
        sys.exit(1)

_check_index(args.llm,          llms,       "llm")
_check_index(args.pro_template, pro_temps,  "pro-template")
if args.mode == "pro-pro":
    _check_index(args.eval_template, pro_temps,  "eval-template")
else:
    _check_index(args.eval_template, perc_temps, "eval-template")

model_name = llms[args.llm]
template1  = pro_temps[args.pro_template]
template2  = (pro_temps if args.mode == "pro-pro" else perc_temps)[args.eval_template]

device = "cuda" if torch.cuda.is_available() else "cpu"
current_model_name = None
tokenizer = None
model = None

def load_model(name):
    global current_model_name, tokenizer, model
    if name == current_model_name:
        print(f"Model {name} already loaded, skipping reload.", file=sys.stderr)
        return
    print(f"Loading {name}...", file=sys.stderr)
    print(f"Using device: {device}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    current_model_name = name

def generate_with_sampling(prompt_text, max_new_tokens=100, temperature=1, top_k=None, top_p=None):
    """
    Manually implement generation loop by:
    1. Running model.forward
    2. Collecting probabilities
    3. Sampling from them
    """
    
    # Get the device of the model's embedding layer (needed for device_map="auto")
    embed_device = model.model.embed_tokens.weight.device
    
    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt_text, return_tensors="pt").to(embed_device)
    prompt_len = prompt_tokens.shape[1]
   
    # Initialize sequence with prompt tokens
    current_ids = prompt_tokens[0].tolist()
    all_probs = []
    all_tokens = []
    all_token_ids = []
    
    # Generation loop
    for step in range(max_new_tokens):
        # Prepare input as tensor on the same device as embeddings
        input_ids = torch.tensor([current_ids], device=embed_device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids)
            # Move logits to CPU and convert to float32 for all subsequent ops
            logits = outputs.logits[0, -1, :].cpu().float()
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Apply top-k filtering
        if top_k is not None:
            top_k_probs, top_k_indices = torch.topk(probs, top_k)
            probs_filtered = torch.zeros_like(probs)
            probs_filtered[top_k_indices] = top_k_probs
            probs = probs_filtered / probs_filtered.sum()
        
        # Apply nucleus (top-p) filtering
        if top_p is not None:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=0)
            sorted_indices_to_remove = cumsum_probs > top_p
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[indices_to_remove] = 0
            probs = probs / probs.sum()
        
        # Sample from distribution
        next_token_id = torch.multinomial(probs, num_samples=1).item()
        next_token_prob = probs[next_token_id].item()
        
        # Get token string
        token_str = tokenizer.decode([next_token_id])
        
        # Store results
        current_ids.append(next_token_id)
        all_token_ids.append(next_token_id)
        all_probs.append(next_token_prob)
        all_tokens.append(token_str)
        
        # Print progress
        #print(f"Step {step}: '{token_str}' (ID: {next_token_id})")
        #print(f"  Probability: {next_token_prob:.6f}")
        
        # Check for EOS token
        if next_token_id == tokenizer.eos_token_id:
            print("\n[EOS token reached]")
            break
    
    # Decode full response
    full_response = tokenizer.decode(current_ids, skip_special_tokens=True)
    
    print("\n" + "-" * 80)
    print(f"Full Response:\n{full_response}\n")
    
    avg_prob = sum(all_probs) / len(all_probs) if all_probs else 0
    print(f"Average probability: {avg_prob:.6f}")
    print(f"Total tokens generated: {len(all_tokens)}")
    
    return {
        "response": full_response,
        "token_ids": current_ids,
        "generated_token_ids": all_token_ids,
        "tokens": all_tokens,
        "probabilities": all_probs,
        "average_prob": avg_prob
    }

def get_token_probabilities(text):
    """
    Goes over a text token by token, and for each token reports
    the probability the model assigns to it given all previous tokens as context.
    """
    
    # Get the device of the model's embedding layer
    embed_device = model.model.embed_tokens.weight.device
    
    # Tokenize the entire text
    tokens = tokenizer.encode(text, return_tensors="pt").to(embed_device)
    num_tokens = tokens.shape[1]
    
    token_probs = []
    token_strings = []
    
    print(f"Text: {text}")
    print(f"Total tokens: {num_tokens}\n")
    print("-" * 80)
    
    # For each token (starting from the second), predict it given the previous tokens
    for i in range(num_tokens - 1):
        # Use all tokens up to position i as context
        context = tokens[:, :i+1]
        
        # Forward pass
        with torch.no_grad():
            outputs = model(context)
            logits = outputs.logits[0, -1, :].cpu().float()
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get the actual next token and its probability
        next_token_id = tokens[0, i+1].item()
        next_token_str = tokenizer.decode([next_token_id])
        token_prob = probs[next_token_id].item()
                
        token_probs.append(token_prob)
        token_strings.append(next_token_str)
 
    avg_prob = sum(token_probs) / len(token_probs) if token_probs else 0

    
    return {
        "tokens": token_strings,
        "probabilities": token_probs,
        "average_prob": avg_prob
    }

def get_continuation_probabilities(prompt_text, continuation_token_ids):
    """
    Given a prompt and a list of token IDs (generated elsewhere),
    return the probability the model assigns to each token
    when it appears after the prompt + all preceding tokens.
    """
    embed_device = model.model.embed_tokens.weight.device
    prompt_tokens = tokenizer.encode(prompt_text, return_tensors="pt").to(embed_device)
    current_ids = prompt_tokens[0].tolist()

    probs_list = []
    for token_id in continuation_token_ids:
        input_ids = torch.tensor([current_ids], device=embed_device)
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :].cpu().float()
        probs = F.softmax(logits, dim=-1)
        probs_list.append(probs[token_id].item())
        current_ids.append(token_id)

    return probs_list


# ── Main loop & TSV output ─────────────────────────────────────────────────────
if args.output:
    output_path = args.output
else:
    output_path = f"llm{args.llm}_{args.mode}_pro{args.pro_template}_eval{args.eval_template}.tsv"

load_model(model_name)

print(f"\nOutput → {output_path}", file=sys.stderr)

outputs_done = 0
with open(output_path, "w", newline="", encoding="utf-8") as out_f:
    writer = csv.writer(out_f, delimiter="\t")

    for zadani in tqdm(vars_list, desc="Tasks", file=sys.stderr):
        if args.max_outputs is not None and outputs_done >= args.max_outputs:
            break
        prompt_gen = template1.replace("{var}", zadani)
        prompt_eval = template2.replace("{var}", zadani)

        # 1) Generate tokens with template 1 → prob1
        result_gen = generate_with_sampling(prompt_gen, max_new_tokens=args.max_tokens)
        gen_token_ids = result_gen["generated_token_ids"]
        gen_tokens = result_gen["tokens"]
        gen_probs = result_gen["probabilities"]

        # Log generated output to stderr
        print(f"Generated: {result_gen['response']}", file=sys.stderr)

        # 2) Evaluate the same tokens under template 2 → prob2
        eval_probs = get_continuation_probabilities(prompt_eval, gen_token_ids)

        # 3) Write TSV – header row for this zadání
        writer.writerow([model_name, template1.replace("\n", "\\n"), template2.replace("\n", "\\n"), zadani])
        # 4) Write per-token rows: token_string, token_id, prob1, prob2
        for tok_str, tok_id, p1, p2 in zip(gen_tokens, gen_token_ids, gen_probs, eval_probs):
            writer.writerow([tok_str.replace("\n", "\\n"), tok_id, f"{p1:.8f}", f"{p2:.8f}"])
        outputs_done += 1

print(f"\nResults written to {output_path}", file=sys.stderr)
print("All done.", file=sys.stderr)