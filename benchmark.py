import numpy as np
from scipy.spatial.distance import cosine
import ollama
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from Levenshtein import distance as levenshtein_distance
import random
from collections import Counter

class CreativityBenchmark:
    def __init__(self, model_name="llama2", embedder_model="all-MiniLM-L6-v2"):
        self.model = model_name
        self.embedder = SentenceTransformer(embedder_model)
        
        self.edit_requests = [
            "make it more humorous",
            "add more suspense",
            "make it more poetic",
            "add a plot twist",
            "change the tone to be more serious",
            "add more descriptive details",
            "change the perspective to first person",
            "add dialogue",
            "make it more concise",
            "add more emotional depth",
            "change the setting",
            "add a new character",
            "change the ending",
            "add more action",
            "make it more mysterious"
        ]
    
    def _generate(self, prompt, temperature=0.7, max_tokens=600):
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": temperature, "max_tokens": max_tokens}
        )
        text = response["response"].strip()
        
        # Remove thinking tags and their content
        while True:
            start = text.lower().find("<think>")
            if start == -1:
                start = text.lower().find("<antthinking>")  # Also check for antthinking tags
            if start == -1:
                break
                
            end = text.lower().find("</think>", start)
            if end == -1:
                end = text.lower().find("</antthinking>", start)  # Check for closing antthinking tag
            if end == -1:
                break
                
            text = text[:start].strip() + " " + text[end + 8:].strip()  # +8 for </think>
            
        return text.strip()

    def free_association(self, max_iter=100, estimate_total=True):
        """Test creative word association with unseen species estimation"""
        print("\n=== Free Association Test ===")
        print(f"Model will generate up to {max_iter} words until repetition occurs.")
        
        words = []
        frequencies = Counter()
        pbar = tqdm(total=max_iter, desc="Free Association")
        
        prompt = " Freely associate lists of words or numbers— just say whatever word next comes to mind. Respond only with one word and nothing else."
        for i in range(max_iter):
            new_word = self._generate(prompt).split()[0].strip('.,!?;:"').lower()
            frequencies[new_word] += 1
            
            print(f"\nWord {i+1}: {new_word}")
            
            if new_word in words:
                print(f"↳ Repetition detected! '{new_word}' was already used.")
                break
            
            words.append(new_word)
            prompt = f"The next word that comes to mind after '{new_word}' is:"
            pbar.update(1)
        
        pbar.close()
        print(f"\nGenerated {len(words)} unique words before first repetition.")
        print("Word sequence:", " → ".join(words))
        
        if estimate_total:
            N = len(words)
            N1 = sum(1 for f in frequencies.values() if f == 1)
            estimated_total = N + (N1 ** 2) / (2 * N)
            print(f"Estimated total vocabulary size: {estimated_total:.0f}")
            return len(words), estimated_total
        
        return len(words)

    def telephone_game(self, seed_text, max_iter=10, similarity_threshold=0.95):
        """Test creative drift through repeated paraphrasing"""
        print("\n=== Telephone Game Test ===")
        print(f"Starting with: {seed_text}")
        
        current = seed_text
        previous = None
        iterations = 0
        
        pbar = tqdm(total=max_iter, desc="Telephone Game")
        for i in range(max_iter):
            print(f"\nIteration {i+1}:")
            
            expanded = self._generate(
                f"Expand this summary into a detailed story:\n{current}",
                temperature=0.8
            )
            print(f"Expanded story:\n{expanded}")
            
            current = self._generate(
                f"Summarize this story in one sentence:\n{expanded}",
                temperature=0.3,
                max_tokens=50
            )
            print(f"New summary: {current}")
            
            if previous is not None:
                edit_sim = 1 - (levenshtein_distance(current, previous) / max(len(current), len(previous)))
                semantic_sim = 1 - cosine(
                    self.embedder.encode(current),
                    self.embedder.encode(previous)
                )
                
                print(f"Edit similarity: {edit_sim:.2f}")
                print(f"Semantic similarity: {semantic_sim:.2f}")
                
                if edit_sim > similarity_threshold and semantic_sim > similarity_threshold:
                    print("\n→ Convergence detected! Stories are too similar.")
                    break
            
            previous = current
            iterations += 1
            pbar.update(1)
        
        pbar.close()
        print(f"\nCompleted {iterations} iterations before convergence.")
        return iterations

    def camels_back(self, seed_text, max_edits=10):
        """Test story coherence under diverse modifications"""
        print("\n=== Camel's Back Test ===")
        print(f"Starting story: {seed_text}")
        
        story = seed_text
        coherent_edits = 0
        
        pbar = tqdm(total=max_edits, desc="Camel's Back")
        for i in range(max_edits):
            print(f"\nEdit Attempt {i+1}:")
            
            num_edits = min(random.randint(1, 3), len(self.edit_requests))
            current_edits = random.sample(self.edit_requests, num_edits)
            print("Requested edits:")
            for edit in current_edits:
                print(f"- {edit}")
            
            edit_prompt = f"Modify this story according to these instructions:\n{story}\n\n"
            edit_prompt += "\n".join(f"- {edit}" for edit in current_edits)
            
            modified_story = self._generate(edit_prompt, temperature=0.8)
            print(f"\nModified story:\n{modified_story}")
            
            check_prompt = f"""Original story: {story}
            Modified story: {modified_story}
            Requested edits:
            {chr(10).join(f'- {edit}' for edit in current_edits)}
            
            Answer these questions with Yes/No:
            1. Is the modified story still coherent and logical?
            2. Were all the requested edits applied successfully?
            3. Is the quality at least as good as the original?"""
            
            check = self._generate(check_prompt, temperature=0.1)
            print(f"\nQuality check results:\n{check}")
            
            # Split into lines and clean up whitespace
            check_lines = [line.strip() for line in check.split('\n') if line.strip()]

            # Look for explicit yes/no answers
            expected_answers = 3  # We're asking 3 questions
            yes_count = 0
            found_answers = 0

            for line in check_lines:
                # Look for numbered responses or lines containing yes/no
                if ('yes' in line or 'no' in line) and any(str(i) in line for i in range(1, expected_answers + 1)):
                    found_answers += 1
                    if 'yes' in line:
                        yes_count += 1

            # Quality check passes if:
            # 1. We found responses to all questions
            # 2. All responses were 'yes'
            if found_answers == expected_answers and yes_count == expected_answers:
                story = modified_story
                coherent_edits += 1
                pbar.update(1)
            else:
                print("→ Quality check failed! Stopping edits.")
                print(f"Found {found_answers} answers, {yes_count} were 'yes'")
                break
                    
                story = modified_story
                coherent_edits += 1
                pbar.update(1)
        
        pbar.close()
        print(f"\nSuccessfully made {coherent_edits} coherent edits.")
        return coherent_edits

    def dont_repeat_yourself(self, template="Write a story about {}", samples=5, min_length=100):
        """Test diversity of outputs with controlled randomness and volume-based scoring
        
        Args:
            template (str): Template string with {} placeholder for concept
            samples (int): Number of stories to generate
            min_length (int): Minimum required length for generated stories
        
        Returns:
            tuple: (volume_score, mean_distance, std_distance, coverage_score)
        """
        print("\n=== Don't Repeat Yourself Test ===")
        
        # Expanded concept list with more variety and structure
        concepts = {
            'sci_fi': ["a time machine", "an alien artifact", "a sentient AI", "a space colony", "a quantum computer"],
            'fantasy': ["an ancient spell book", "a magical ring", "an enchanted forest", "a dragon's lair", "a wizard's tower"],
            'mystery': ["a mysterious door", "a cursed mirror", "a hidden passage", "an encrypted message", "a detective's journal"],
            'historical': ["a lost civilization", "a forgotten prophecy", "an ancient map", "a royal tomb", "a legendary sword"]
        }
        
        generations = []
        embeddings = []
        categories_used = set()
        
        pbar = tqdm(total=samples, desc="Testing Output Diversity")
        
        # Ensure we use different categories when possible
        for i in range(samples):
            # Select category with lowest usage
            available_categories = [cat for cat in concepts.keys() 
                                if len(categories_used) < len(concepts) or cat in categories_used]
            category = min(available_categories, key=lambda x: categories_used.count(x))
            categories_used.add(category)
            
            # Select random concept from category
            concept = random.choice(concepts[category])
            prompt = template.format(concept)
            
            print(f"\nGeneration {i+1} ({category}):")
            print(f"Prompt: {prompt}")
            
            text = self._generate(prompt)
            
            # Validate output length
            if len(text.split()) < min_length:
                print(f"Warning: Generation {i+1} is shorter than minimum length")
            
            print(f"Generated story ({len(text.split())} words):\n{text}")
            
            generations.append(text)
            embeddings.append(self.embedder.encode(text))
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
        
        return volume_score, mean_dist, std_dist, coverage_score

    def combined_score(self, seed_text, weights=None):
        """Calculate overall creativity score using multiple metrics"""
        print("\n====== Running Creativity Benchmark Suite ======")
        print(f"Model: {self.model}")
        print(f"Seed text: {seed_text}")
        print("================================================")
        
        # Run all tests
        fa_score, fa_total = self.free_association()
        tel_score = self.telephone_game(seed_text)
        cb_score = self.camels_back(seed_text)
        div_mean, div_std = self.dont_repeat_yourself()
        
        scores = {
            "free_association": fa_score,
            "estimated_vocabulary": fa_total,
            "telephone_game": tel_score,
            "camels_back": cb_score,
            "diversity_mean": div_mean,
            "diversity_std": div_std
        }
        
        # Normalize scores (higher = better)
        normalized = {
            "free_association": fa_score / 100,
            "telephone_game": tel_score / 10,
            "camels_back": cb_score / 10,
            "diversity": div_mean
        }
        
        # Default equal weights
        weights = weights or {k: 1/len(normalized) for k in normalized}
        
        composite = sum(normalized[k] * weights[k] for k in normalized)
        return {
            "scores": scores,
            "normalized": normalized,
            "composite": composite
        }

# Example usage
if __name__ == "__main__":
    benchmark = CreativityBenchmark(model_name="deepseek-r1:1.5b")
    results = benchmark.combined_score("A dragon guarded a treasure.")
    
    print("\n============= Final Results =============")
    print(f"Composite Creativity Score: {results['composite']:.2f}")
    print("\nRaw Scores:")
    for k, v in results["scores"].items():
        if isinstance(v, float):
            print(f"- {k}: {v:.2f}")
        else:
            print(f"- {k}: {v}")
    
    print("\nNormalized Scores:")
    for k, v in results["normalized"].items():
        print(f"- {k}: {v:.2f}")
    print("=========================================")