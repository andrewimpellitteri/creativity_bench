from tqdm.auto import tqdm
from Levenshtein import distance as levenshtein_distance
import ollama
from scipy.spatial.distance import cosine

class TelephoneGameMixin:
    def telephone_game(self, seed_text, max_iter=10, percentile_threshold=0.9, min_samples=3):
        """Test creative drift through repeated paraphrasing using dynamic thresholds"""
        print("\n=== Telephone Game Test ===")
        print(f"Starting with: {seed_text}")
        
        if not seed_text.strip():
            raise ValueError("Seed text cannot be empty")
            
        current = seed_text
        previous = None
        iterations = 0
        edit_similarities = []
        semantic_similarities = []
        
        pbar = tqdm(total=max_iter, desc="Telephone Game")
        for i in range(max_iter):
            print(f"\nIteration {i+1}:")
            
            expanded = self._generate(
                f"Expand this summary into a detailed story:\n{current}",
                temperature=0.8
            )
            if not expanded.strip():
                raise ValueError(f"Empty expanded story in iteration {i+1}")
            print(f"Expanded:\n{expanded}")
            
            current = self._generate(
                f"Summarize this story in one sentence:\n{expanded}",
                temperature=0.3
            )
            if not current.strip():
                raise ValueError(f"Empty summary in iteration {i+1}")
            print(f"New summary: {current}")
            
            if previous:
                # Calculate similarities
                edit_sim = 1 - (levenshtein_distance(current, previous) / max(len(current), len(previous)))
                semantic_sim = 1 - cosine(
                    ollama.embeddings(model='nomic-embed-text', prompt=current).embedding,
                    ollama.embeddings(model='nomic-embed-text', prompt=previous).embedding
                )
                
                # Store similarities
                edit_similarities.append(edit_sim)
                semantic_similarities.append(semantic_sim)
                
                print(f"Edit similarity: {edit_sim:.2f}")
                print(f"Semantic similarity: {semantic_sim:.2f}")
                
                # Check convergence after collecting enough samples
                if len(edit_similarities) >= min_samples:
                    # Calculate percentile ranks
                    edit_rank = sum(es <= edit_sim for es in edit_similarities) / len(edit_similarities)
                    semantic_rank = sum(ss <= semantic_sim for ss in semantic_similarities) / len(semantic_similarities)
                    
                    # Check if both metrics exceed percentile threshold
                    if edit_rank >= percentile_threshold and semantic_rank >= percentile_threshold:
                        print(f"\n→ Convergence detected (top {percentile_threshold*100:.0f}% similarity)")
                        break
            
            previous = current
            iterations += 1
            pbar.update(1)
        
        pbar.close()
        print(f"\nCompleted {iterations} iterations before convergence.")
        return iterations
