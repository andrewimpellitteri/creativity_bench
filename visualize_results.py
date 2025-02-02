import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def load_all_json_data():
    """Load all JSON files from runs/ directory and aggregate data by model name"""
    data = {}
    for json_file in glob.glob('runs/*.json'):
        with open(json_file, 'r') as f:
            file_data = json.load(f)
            for model, info in file_data.items():
                if model not in data:
                    # Create a container for all metrics for this model
                    data[model] = {
                        'composites': [],
                        'normalized': [],
                        'scores': []
                    }
                data[model]['composites'].append(info['composite'])
                data[model]['normalized'].append(info['normalized'])
                data[model]['scores'].append(info['scores'])
    return data

def process_data(data):
    """Aggregate data per model and extract average composite score, error, and diversity"""
    models = []
    composites = []
    errors = []
    diversities = []

    for k, v in data.items():
        print(k)
        print(v)
    
    for model, metrics in data.items():
        models.append(model)
        # Average composite scores for models that appear multiple times
        avg_composite = np.mean(metrics['composites'])
        composites.append(avg_composite)

        errors.append(np.std(metrics['composites']))
        
        # For diversity, check if the 'diversity' key exists and average across all instances
        diversity_values = [norm.get('diversity', 0) for norm in metrics['normalized'] if 'diversity' in norm]
        diversity = np.mean(diversity_values) if diversity_values else 0
        diversities.append(diversity)

        # Debug: print the aggregated info per model
        print(f"Model: {model}, avg composite: {avg_composite}, error: {errors}, diversity: {diversity}")
    
    return models, composites, errors, diversities

def plot_composite_scores(models, composites, errors, diversities):
    """Create bar chart with error bars and diversity indicators"""
    plt.figure(figsize=(12, 7))
    x = np.arange(len(models))
    width = 0.6  # Bar width
    
    bars = plt.bar(
        x,
        composites,
        width,
        yerr=errors,
        capsize=8,
        alpha=0.8,
        error_kw={'elinewidth': 2, 'ecolor': 'black'}
    )
    
    plt.ylabel('Composite Score', fontsize=12)
    plt.title('Model Creativity Performance Comparison', fontsize=14)
    plt.xticks(x, models, rotation=35, ha='right', fontsize=10)
    plt.ylim(0, max(composites) * 1.15)
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.2f}',
            ha='center',
            va='bottom'
        )
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # Load and process all JSON files from the runs directory
    aggregated_data = load_all_json_data()
    
    if not aggregated_data:
        print("No JSON files found in runs/ directory!")
    else:
        models, composites, errors, diversities = process_data(aggregated_data)
        plot_composite_scores(models, composites, errors, diversities)
        print(f"Generated comparison plot with {len(models)} models")
