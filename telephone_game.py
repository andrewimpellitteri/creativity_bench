from tqdm.auto import tqdm
from jellyfish import damerau_levenshtein_distance
import ollama
from scipy.spatial.distance import cosine
from rouge import Rouge
import numpy as np
import json
import os

class TelephoneGameMixin:
    def _write_seq_to_file(self, semantic, edit):
        # Create a dictionary with your desired structure
        new_data = {self.model: {"semantic": semantic, "edit": edit}}


        fname = "sequences.json"
        
        # Check if the file exists
        if os.path.exists(fname):
            # If it exists, read the current data
            with open(fname, 'r') as file:
                existing_data = json.load(file)
            
            # Append the new data to the existing data
            existing_data.append(new_data)
            
            # Write the updated data back to the file
            with open(fname, 'w') as file:
                json.dump(existing_data, file, indent=4)
        else:
            # If the file doesn't exist, create a new file with the data
            with open(fname, 'w') as file:
                json.dump([new_data], file, indent=4)

    def _normalized_damerau_levenshtein_similarity(self, hypothesis, reference):
        edit_distance = damerau_levenshtein_distance(hypothesis, reference)
        max_len = max(len(hypothesis), len(reference))
        return 1 - (edit_distance / max_len)

    def _get_rouge_l(self, hypothesis, reference):
        rouge = Rouge()
        scores = rouge.get_scores(hypothesis, reference)
        return scores[0]["rouge-l"]["f"]

    def _combined_edit_distance_score(self, hypothesis, reference):
        rouge_score = self._get_rouge_l(hypothesis, reference)
        dl_sim_score = self._normalized_damerau_levenshtein_similarity(
            hypothesis, reference
        )

        return (rouge_score + dl_sim_score) / 2

    def telephone_game(
        self, seed_text, max_iter=10, percentile_threshold=0.8, min_samples=3
    ):
        """Test creative drift through repeated paraphrasing using dynamic thresholds"""
        print("\n=== Telephone Game Test ===")
        print(f"Starting with: {seed_text}")
        print(self.model)

        d_thres = 0.01

        if not seed_text.strip():
            raise ValueError("Seed text cannot be empty")

        current = seed_text
        previous = None
        iterations = 0
        edit_similarities = []
        semantic_similarities = []

        pbar = tqdm(total=max_iter, desc="Telephone Game")
        for i in range(max_iter):
            print(f"\nIteration {i + 1}:")

            expanded = self._generate(
                f"Expand this summary into a detailed story:\n{current}",
                temperature=0.8,
            )

            print(rf"Raw expanded {expanded}")
            if not expanded.strip():
                raise ValueError(f"Empty expanded story in iteration {i + 1}")
            print(f"Expanded:\n{expanded}")

            current = self._generate(
                f"Summarize this story in one sentence:\n{expanded}", temperature=0.3
            )
            if not current.strip():
                raise ValueError(f"Empty summary in iteration {i + 1}")
            print(f"New summary: {current}")

            if previous:
                # Calculate similarities
                edit_sim = 1 - self._combined_edit_distance_score(current, previous)
                semantic_sim = 1 - cosine(
                    ollama.embeddings(
                        model="nomic-embed-text", prompt=current
                    ).embedding,
                    ollama.embeddings(
                        model="nomic-embed-text", prompt=previous
                    ).embedding,
                )

                # Store similarities
                edit_similarities.append(edit_sim)
                semantic_similarities.append(semantic_sim)

                print(f"Edit similarity: {edit_sim:.2f}")
                print(f"Semantic similarity: {semantic_sim:.2f}")

                # Check convergence after collecting enough samples
                if len(edit_similarities) >= min_samples:
                    # Calculate percentile ranks
                    # edit_rank = sum(es <= edit_sim for es in edit_similarities) / len(edit_similarities)
                    # semantic_rank = sum(ss <= semantic_sim for ss in semantic_similarities) / len(semantic_similarities)

                    print(f"Edit similarities: {edit_similarities}")
                    print(f"Semantic similarities: {semantic_similarities}")

                    edit_derivatives = np.gradient(edit_similarities)
                    semantic_derivatives = np.gradient(semantic_similarities)

                    # Check if both metrics exceed percentile threshold
                    # if edit_rank >= percentile_threshold and semantic_rank >= percentile_threshold:
                    #     print(f"\n→ Convergence detected (top {percentile_threshold*100:.0f}% similarity)")
                    #     break
                    print(np.abs(edit_derivatives), np.abs(semantic_derivatives))
                    if np.all(np.abs(edit_derivatives) < d_thres) and np.all(
                        np.abs(semantic_derivatives) < d_thres
                    ):
                        print("Convergence detected: derivatives are approaching zero.")
                    else:
                        print("Derivatives have not yet approached zero.")

            previous = current
            iterations += 1
            pbar.update(1)

        pbar.close()
        self._write_seq_to_file(edit_similarities, semantic_similarities)
        print(f"\nCompleted {iterations} iterations before convergence.")
        return iterations
