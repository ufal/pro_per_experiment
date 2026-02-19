import sys
import os
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


def read_datafile(path):
    """
    Read a TSV datafile with the following format:
      Line 1: model_name
      Remaining lines (TSV): template1 \t template2 \t zadani
    template1 may contain '{var}' which will be replaced by zadani.
    Returns (model_name, list of (template1, template2, zadani) tuples).
    """
    with open(path, encoding="utf-8") as f:
        model_name = f.readline().strip()
        reader = csv.reader(f, delimiter="\t")
        data = []
        for row in reader:
            if len(row) < 3:
                continue  # skip malformed / empty lines
            template1, template2, zadani = row[0], row[1], row[2]
            data.append((template1, template2, zadani))
    return model_name, data


# ── Read datafile ──────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python experiment.py <datafile.tsv> [output.tsv]")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(input_path)[0] + "_output.tsv"
model_name, data_file = read_datafile(input_path)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {model_name}...")
print(f"Using device: {device}\n")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
)

if device == "cpu":
    model = model.to(device)

model.eval()

def generate_with_sampling(prompt_text, max_new_tokens=50, temperature=1, top_k=None, top_p=None):
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
    
    print(f"Prompt: {prompt_text}")
    print(f"Initial tokens: {prompt_len}\n")
    print("-" * 80)
    
    # Initialize sequence with prompt tokens
    current_ids = prompt_tokens[0].tolist()
    all_probs = []
    all_tokens = []
    all_token_ids = []
    
    # Generation loop
    for step in tqdm(range(max_new_tokens), desc="Generating", file=sys.stderr):
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
        print(f"Step {step}: '{token_str}' (ID: {next_token_id})")
        print(f"  Probability: {next_token_prob:.6f}")
        
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
    for i in tqdm(range(num_tokens - 1), desc="Token probs", file=sys.stderr):
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
        
        # Current context token for display
        current_token_str = tokenizer.decode([tokens[0, i].item()])
        
        token_probs.append(token_prob)
        token_strings.append(next_token_str)
        
        print(f"Position {i}: context ends with '{current_token_str}'")
        print(f"  → Next token: '{next_token_str}' | Probability: {token_prob:.6f}")
    
    print("\n" + "-" * 80)
    print(f"\nSummary:")
    for tok, prob in zip(token_strings, token_probs):
        print(f"  '{tok}': {prob:.6f}")
    
    avg_prob = sum(token_probs) / len(token_probs) if token_probs else 0
    print(f"\nAverage probability: {avg_prob:.6f}")
    
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
    for token_id in tqdm(continuation_token_ids, desc="Eval probs", file=sys.stderr):
        input_ids = torch.tensor([current_ids], device=embed_device)
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :].cpu().float()
        probs = F.softmax(logits, dim=-1)
        probs_list.append(probs[token_id].item())
        current_ids.append(token_id)

    return probs_list


# ── Main loop & TSV output ─────────────────────────────────────────────────────
with open(output_path, "w", newline="", encoding="utf-8") as out_f:
    writer = csv.writer(out_f, delimiter="\t")

    for template1, template2, zadani in tqdm(data_file, desc="Tasks", file=sys.stderr):
        prompt_gen = template1.replace("{var}", zadani)
        prompt_eval = template2.replace("{var}", zadani)

        print(f"\n{'=' * 80}")
        print(f"Zadání: {zadani}")
        print(f"Generation prompt: {prompt_gen}")
        print(f"Evaluation prompt: {prompt_eval}")
        print(f"{'=' * 80}\n")

        # 1) Generate tokens with template 1 → prob1
        result_gen = generate_with_sampling(prompt_gen)
        gen_token_ids = result_gen["generated_token_ids"]
        gen_tokens = result_gen["tokens"]
        gen_probs = result_gen["probabilities"]

        # 2) Evaluate the same tokens under template 2 → prob2
        eval_probs = get_continuation_probabilities(prompt_eval, gen_token_ids)

        # 3) Write TSV – header row for this zadání
        writer.writerow([model_name, template1, template2, zadani])
        # 4) Write per-token rows: token_string, token_id, prob1, prob2
        for tok_str, tok_id, p1, p2 in zip(gen_tokens, gen_token_ids, gen_probs, eval_probs):
            writer.writerow([tok_str, tok_id, f"{p1:.8f}", f"{p2:.8f}"])

print(f"\nResults written to {output_path}")