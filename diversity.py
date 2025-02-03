import numpy as np
from tqdm.auto import tqdm
import random
from scipy.spatial.distance import cosine
import ollama
from collections import Counter

class DiversityMixin:
    def dont_repeat_yourself(self, template="Write a story about {}", samples=5, min_length=100):
        """Test diversity of outputs with controlled randomness and volume-based scoring"""
        print("\n=== Don't Repeat Yourself Test ===")
        
        concepts = {
            'sci_fi': ["a time machine", "an alien artifact", "a sentient AI", "a space colony", "a quantum computer"],
            'fantasy': ["an ancient spell book", "a magical ring", "an enchanted forest", "a dragon's lair", "a wizard's tower"],
            'mystery': ["a mysterious door", "a cursed mirror", "a hidden passage", "an encrypted message", "a detective's journal"],
            'historical': ["a lost civilization", "a forgotten prophecy", "an ancient map", "a royal tomb", "a legendary sword"]
        }
        
        generations = []
        embeddings = []
        categories_used = []  # Changed from set to list
        
        pbar = tqdm(total=samples, desc="Testing Output Diversity")
        
        for i in range(samples):
            # Select category with lowest usage count
            available_categories = list(concepts.keys())  # Always consider all categories
            category = min(available_categories, key=lambda x: categories_used.count(x))
            categories_used.append(category)  # Append to list
            
            concept = random.choice(concepts[category])
            prompt = template.format(concept)
            
            print(f"\nGeneration {i+1} ({category}):")
            print(f"Prompt: {prompt}")
            
            text = self._generate(prompt)
            
            if len(text.split()) < min_length:
                print(f"Warning: Generation {i+1} is shorter than minimum length")
            
            print(f"Generated story ({len(text.split())} words):\n{text}")
            
            generations.append(text)
            embeddings.append(ollama.embeddings(model='nomic-embed-text', prompt=text).embedding)
            pbar.update(1)
        
        pbar.close()
        
        # Calculate pairwise distances and similarities
        distances = []
        similarities = np.zeros((samples, samples))
        print("\nPairwise similarities:")
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = cosine(embeddings[i], embeddings[j])
                sim = 1 - dist
                distances.append(dist)
                similarities[i,j] = similarities[j,i] = sim
                print(f"Story {i+1} vs Story {j+1}: {sim:.2f} similarity")
        
        # Calculate volume score using determinant of similarity matrix
        volume_score = np.linalg.det(similarities + np.eye(samples))
        
        # Calculate coverage score (how well we used different categories)
        category_counts = Counter(categories_used)
        expected_count = samples / len(concepts)
        coverage_score = 1 - np.std([count/expected_count for count in category_counts.values()])
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        print(f"\nVolume score: {volume_score:.3f}")
        print(f"Mean diversity (distance): {mean_dist:.3f}")
        print(f"Standard deviation: {std_dist:.3f}")
        print(f"Category coverage score: {coverage_score:.3f}")
        
        # Additional analysis
        print("\nDiversity Analysis:")
        print(f"- Most similar pair: Stories {np.unravel_index(similarities.argmax(), similarities.shape)[0]+1} "
            f"and {np.unravel_index(similarities.argmax(), similarities.shape)[1]+1}")
        print(f"- Least similar pair: Stories {np.unravel_index(similarities.argmin(), similarities.shape)[0]+1} "
            f"and {np.unravel_index(similarities.argmin(), similarities.shape)[1]+1}")
        
        return volume_score