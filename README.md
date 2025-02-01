# LLM Creativity Benchmark
A comprehensive evaluation suite for measuring creative capabilities in large language models (LLMs). This benchmark assesses multiple key dimensions of creativity through structured tests and quantitative metrics.

Based on [this](https://gwern.net/creative-benchmark) post by Gwern.

## Key Features
- **Free Association Test**: Measures lexical originality and vocabulary estimation
- **Telephone Game**: Quantifies semantic drift through iterative paraphrasing
- **Camel's Back Challenge**: Tests narrative coherence under multiple edits
- **DRY (Don't Repeat Yourself) Test**: Evaluates output diversity across prompts
- **Extreme Style Transfer Test**: Take a set of stories with genre labels; ask a LLM to summarize each one; then ask it to write a story using only the summary and a random other genre label; score based on how different the other genre versions are from the original. 
- **Composite Creativity Score**: Combined metric aggregating multiple dimensions

Currently only supports `ollama` for local generation but will update to work OpenAI API and others soon.

## Installation and Usage

Clone the repository and run:

```
pip install -r requirements.txt
```

To use:

```
usage: benchmark.py [-h] [--model MODEL] [--prompt PROMPT] [--save]

Run the LLM Creativity Benchmark and output the results.

options:
  -h, --help       show this help message and exit
  --model MODEL    Name of the model to benchmark.
  --prompt PROMPT  Input prompt for the benchmark (e.g., 'A dragon guarded a treasure.')
  --save           If set, save the results as a JSON file in the 'runs' directory.
```

## Contributing

Please feel free to contribute by submitting pull requests or issues. We welcome any feedback on how we can improve the benchmark suite.

## License

MIT