# LLM Creativity Benchmark

A comprehensive evaluation suite for measuring creative capabilities in large language models (LLMs). This benchmark assesses four key dimensions of creativity through structured tests and quantitative metrics.

Based on [this](https://gwern.net/creative-benchmark) post by Gwern.

## Key Features
- **Free Association Test**: Measures lexical originality and vocabulary estimation
- **Telephone Game**: Quantifies semantic drift through iterative paraphrasing
- **Camel's Back Challenge**: Tests narrative coherence under multiple edits
- **DRY (Don't Repeat Yourself) Test**: Evaluates output diversity across prompts
- **Extreme Style Transfer Test**: Take a set of stories with genre labels; ask a LLM to summarize each one; then ask it to write a story using only the summary and a random other genre label; score based on how different the other genre versions are from the original. 
- **Composite Creativity Score**: Combined metric aggregating multiple dimensions

## Installation and Usage

Clone the repository and run:

```
pip install -r requirements.txt
```

and then

```
python benchmark.py
```
