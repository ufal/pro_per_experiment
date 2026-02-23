# Production vs. Perception

Compares token-level probabilities of LLM-generated text under two different prompt templates. The model generates a continuation using **template 1** (production), then the same tokens are evaluated under **template 2** (perception), yielding paired per-token probability estimates.

## Usage

### Input format

Plain text files with 3 header lines followed by one item per line:

```
meta-llama/Llama-3.1-8B
User: Please write a poem about {var}.\n\nAssistant: Here is the poem for you:\n\n
User: Here is a poem about {var}, please rate the poem.\n\n
a hedgehog
an octopus
a whale
```

- **Line 1:** HuggingFace model name
- **Line 2:** Generation template (`{var}` is replaced by each item, `\n` becomes a newline)
- **Line 3:** Evaluation template
- **Remaining lines:** Items to iterate over

### Running the experiment

```bash
python experiment.py input.tsv                # output → input_output.tsv
python experiment.py input.tsv -o results.tsv # custom output path
python experiment.py a.tsv b.tsv              # multiple input files
python experiment.py input.tsv --max-tokens 200 --max-outputs 10
```

### Output format

For each item, the output TSV contains a header row followed by per-token rows:

```
model_name	template1	template2	item
token_string	token_id	prob_template1	prob_template2
token_string	token_id	prob_template1	prob_template2
...
```

### Generating pairwise combinations

```bash
python add_combinations.py input.tsv
```

Appends all ordered pairs (e.g. "a hedgehog and an octopus") to the input file.

## Requirements

- Python 3
- PyTorch, Transformers, NumPy, tqdm
