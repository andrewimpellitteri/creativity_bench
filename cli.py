import argparse
import json
import os
import uuid
from main import CreativityBenchmark

def print_results(results):
    output_lines = []
    output_lines.append("\n============= Final Results =============")
    output_lines.append(f"Composite Creativity Score: {results['composite']:.2f}")
    
    output_lines.append("\nRaw Scores:")
    for k, v in results["scores"].items():
        if isinstance(v, float):
            output_lines.append(f"- {k}: {v:.2f}")
        else:
            output_lines.append(f"- {k}: {v}")
    
    output_lines.append("\nNormalized Scores:")
    for k, v in results["normalized"].items():
        output_lines.append(f"- {k}: {v:.2f}")
    output_lines.append("=========================================")
    
    # Join and return the output string
    return "\n".join(output_lines)

def main():
    parser = argparse.ArgumentParser(
        description="Run the LLM Creativity Benchmark and output the results."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="qwen2.5:0.5b", 
        help="Name of the model to benchmark."
    )
    parser.add_argument(
        "--prompt", 
        type=str,
        default="A dragon guarded a treasure.",
        help="Input prompt for the benchmark"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results as JSON file in 'runs' directory"
    )
    parser.add_argument(
        "--use_api",
        action="store_true",
        help="Use Hugging Face API for generation"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of benchmark runs"
    )

    args = parser.parse_args()

    print(f"Running benchmark {args.n} times...")
    for _ in range(args.n):
        benchmark = CreativityBenchmark(
            model_name=args.model, 
            use_api=args.use_api
        )
        results = benchmark.combined_score(args.prompt)

        print_results(results)
        
        if args.save:
            run_id = uuid.uuid4().hex
            output_dir = "runs"
            os.makedirs(output_dir, exist_ok=True)
            
            safe_model_name = "".join(
                c if c.isalnum() or c in "-_" else "_" 
                for c in args.model
            )
            filename = f"{safe_model_name}_{run_id}.json"
            output_path = os.path.join(output_dir, filename)
            
            try:
                with open(output_path, "w") as f:
                    json.dump({args.model: results}, f, indent=4)
                print(f"Results saved to '{output_path}'")
            except Exception as e:
                print(f"Error saving results: {e}")

if __name__ == "__main__":
    main()