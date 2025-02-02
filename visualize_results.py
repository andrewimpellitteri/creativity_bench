import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def load_all_json_data():
    """Load all JSON files from current directory"""
    data = {}
    for json_file in glob.glob('runs/*.json'):
        with open(json_file, 'r') as f:
            file_data = json.load(f)
            data.update(file_data)
    return data

def process_data(data):
    """Process data to extract metrics"""
    models = []
    composites = []
    errors = []
    diversities = []
    
    for model, info in data.items():
        models.append(model)
        composites.append(info['composite'])
        
        # Calculate error from normalized scores
        normalized = list(info['normalized'].values())
        errors.append(np.std(normalized))
        
        # Get diversity score if available
        diversities.append(info['normalized'].get('diversity', 0))
        
    return models, composites, errors, diversities

def plot_composite_scores(models, composites, errors, diversities):
    """Create bar chart with error bars and diversity indicators"""
    plt.figure(figsize=(12, 7))
    x = np.arange(len(models))
    width = 0.6  # Bar width
    
    bars = plt.bar(x, composites, width, yerr=errors, 
                  capsize=8, alpha=0.8, error_kw={'elinewidth': 2})
    
    # Add diversity indicators
    for i, div in enumerate(diversities):
        plt.plot(x[i], div * composites[i], 'ro', markersize=8)
    
    plt.ylabel('Composite Score', fontsize=12)
    plt.title('Model Performance Comparison\n(Red dots show diversity score proportion)', fontsize=14)
    plt.xticks(x, models, rotation=35, ha='right', fontsize=10)
    plt.ylim(0, max(composites) * 1.15)
    
    # Add grid and value labels
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # Load and process all JSON files
    data = load_all_json_data()
    
    if not data:
        print("No JSON files found in current directory!")
    else:
        models, composites, errors, diversities = process_data(data)
        plot_composite_scores(models, composites, errors, diversities)
        print(f"Generated comparison plot with {len(models)} models")