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