import numpy as np
from scipy.spatial.distance import cosine
import ollama
from tqdm.auto import tqdm
from Levenshtein import distance as levenshtein_distance
import random
from collections import Counter
from utils import sample_stories, genre_list
import argparse
import json
import os
import uuid
import requests

class CreativityBenchmark:
    def __init__(self, model_name, use_api):
        self.model = model_name
        self.hf_token = os.getenv("HF_TOKEN")
        self.use_api = use_api

        
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

        if self.use_api:
            if not self.hf_token:
                raise ValueError("Hugging Face API token not provided. Set it during initialization or via HF_API_TOKEN environment variable.")

            # Hugging Face API endpoint
            API_URL = f"https://api-inference.huggingface.co/models/{self.model}"
            
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "return_full_text": False  # Only return the generated text
                }
            }
            
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code != 200:
                raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
                
            # Extract generated text from response
            try:
                text = response.json()[0]["generated_text"]
            except (KeyError, IndexError) as e:
                raise Exception(f"Unexpected API response format: {response.text}") from e
        else:  
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": temperature, "max_tokens": max_tokens}
            )
            
            text = response["response"].strip()
        

        start = text.lower().find("<think>")
        if start == -1:
            start = text.lower().find("<antthinking>")  # Also check for antthinking tags
            
        end = text.lower().find("</think>", start)
        if end == -1:
            end = text.lower().find("</antthinking>", start)  # Check for closing antthinking tag

            
        text = text[:start].strip() + " " + text[end + 8:].strip()  # +8 for </think>
            
        return text.strip()

    def free_association(self, max_iter=100, estimate_total=True):
        """Test creative word association with unseen species estimation"""
        print("\n=== Free Association Test ===")
        print(f"Model will generate up to {max_iter} words until repetition occurs.")
        
        words = []
        frequencies = Counter()
        pbar = tqdm(total=max_iter, desc="Free Association")
        
        prompt = "Freely associate lists of words or numbers— just say whatever word next comes to mind. Respond only with one word and nothing else."
        for i in range(max_iter):
            new_word = self._generate(prompt).split()[0].strip('.,!?;:*"').lower()
            frequencies[new_word] += 1
            
            print(f"\nWord {i+1}: {new_word}")
            
            if new_word in words:
                print(f"↳ Repetition detected! '{new_word}' was already used.")
                break
            
            words.append(new_word)
            prompt = f"Respond with only one word and nothing else. The next word that comes to mind after '{new_word}' is:"
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

    def telephone_game(self, seed_text, max_iter=10, semantic_similarity_threshold=0.80, edit_similarity_threshold=0.30):
        """Test creative drift through repeated paraphrasing"""
        print("\n=== Telephone Game Test ===")
        print(f"Starting with: {seed_text}")
        
        if not seed_text or not seed_text.strip():
            raise ValueError("Seed text cannot be empty")
            
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

            print(expanded)
            
            # Validate expanded response
            if not expanded or not expanded.strip():
                raise ValueError(f"Received empty expanded story in iteration {i+1}")
                
            print(f"Expanded story:\n{expanded}")
            
            current = self._generate(
                prompt=f"Summarize this story in one sentence:\n{expanded}",
                temperature=0.3
            )
            
            # Validate summary response
            if not current or not current.strip():
                raise ValueError(f"Received empty summary in iteration {i+1}")
                
            print(f"New summary: {current}")
            
            if previous is not None:
                edit_sim = 1 - (levenshtein_distance(current, previous) / max(len(current), len(previous)))
                semantic_sim = 1 - cosine(
                    ollama.embeddings(model='nomic-embed-text', prompt=current).embedding,
                    ollama.embeddings(model='nomic-embed-text', prompt=previous).embedding
                )
                
                print(f"Edit similarity: {edit_sim:.2f}")
                print(f"Semantic similarity: {semantic_sim:.2f}")
                
                if edit_sim > edit_similarity_threshold and semantic_sim > semantic_similarity_threshold:
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
            
            Answer these questions with Yes/No and do not respond with anything else:
            1. Is the modified story still coherent and logical?
            2. Were all the requested edits applied successfully?
            3. Is the quality at least as good as the original?"""
            
            check = self._generate(check_prompt, temperature=0.1)
            print(f"\nQuality check results:\n{check}")
            
            # Split into lines and clean up whitespace
            check_lines = [line.strip().lower() for line in check.split('\n') if line.strip()]

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

    def extreme_style_transfer(self, stories, genres,
                               summary_prompt_template="Summarize the following story:\n\n{}",
                               style_prompt_template="Using only the summary below and the target genre '{}', write a new story:\n\nSummary: {}\n",
                               min_summary_length=20):
        """
        Extreme Style Transfer Benchmark

        Parameters:
            stories (list of dict): Each dict should have keys:
                                    - 'genre': the original genre label (str)
                                    - 'text': the original story text (str)
            genres (list of str): List of possible genre labels.
            summary_prompt_template (str): Template to prompt the LLM for summarization.
            style_prompt_template (str): Template to prompt the LLM for generating a style-transferred story.
            min_summary_length (int): Minimum word count expected for the summary.

        Process:
            1. For each story, ask the LLM to generate a summary.
            2. Choose a random genre label from the provided list that is different from the original.
            3. Ask the LLM to write a new story using only the summary and the target genre.
            4. Compute a difference score based on the cosine distance between embeddings of the original story and the transformed story.
        """
        original_texts = []
        summaries = []
        transferred_texts = []
        transfer_scores = []

        print("\n=== Extreme Style Transfer Benchmark ===")
        pbar = tqdm(total=len(stories), desc="Processing Stories")

        for story_dict in stories:
            orig_genre = story_dict['genre']
            orig_text = story_dict['text']
            original_texts.append(orig_text)

            # Step 1: Summarize the original story.
            summary_prompt = summary_prompt_template.format(orig_text)
            summary = self._generate(summary_prompt)
            # Check for summary length; warn if too short
            if len(summary.split()) < min_summary_length:
                print("Warning: The generated summary is shorter than the minimum length required.")
            summaries.append(summary)

            # Step 2: Choose a random target genre that is different from the original.
            available_target_genres = [g for g in genres if g != orig_genre]
            if not available_target_genres:
                raise ValueError("No alternate genre available for style transfer.")
            target_genre = random.choice(available_target_genres)

            # Step 3: Generate a new story based on the summary and the target genre.
            style_prompt = style_prompt_template.format(target_genre, summary)
            transferred_text = self._generate(style_prompt)
            transferred_texts.append(transferred_text)

            # Step 4: Compute the difference score between the original and transferred stories.
            # Here we assume the existence of an embedding API via ollama.embeddings.
            orig_embedding = ollama.embeddings(model='nomic-embed-text', prompt=orig_text).embedding
            transferred_embedding = ollama.embeddings(model='nomic-embed-text', prompt=transferred_text).embedding
            score = cosine(orig_embedding, transferred_embedding)  # cosine distance; higher means more different
            transfer_scores.append(score)

            pbar.update(1)

        pbar.close()

        # Compute aggregate statistics.
        mean_score = np.mean(transfer_scores)
        std_score = np.std(transfer_scores)

        print("\n=== Extreme Style Transfer Results ===")
        for idx, score in enumerate(transfer_scores):
            print(f"Story {idx+1}: Cosine distance = {score:.3f}")

        print(f"\nAverage difference score (mean cosine distance): {mean_score:.3f}")
        print(f"Standard deviation of scores: {std_score:.3f}")

        return mean_score


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
        volume_score = self.dont_repeat_yourself()
        extreme_score = self.extreme_style_transfer(sample_stories, genre_list)
        
        scores = {
            "free_association": fa_score,
            "estimated_vocabulary": fa_total,
            "telephone_game": tel_score,
            "camels_back": cb_score,
            "dont_repeat_yourself": volume_score,
            "extreme_style_transfer": extreme_score
        }
        
        # Normalize scores (higher = better)
        normalized = {
            "free_association": fa_score / 100,
            "telephone_game": tel_score / 10,
            "camels_back": cb_score / 10,
            "diversity": volume_score
        }
        
        # Default equal weights
        weights = weights or {k: 1/len(normalized) for k in normalized}
        
        composite = sum(normalized[k] * weights[k] for k in normalized)
        return {
            "scores": scores,
            "normalized": normalized,
            "composite": composite
        }

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
        help="Input prompt for the benchmark (e.g., 'A dragon guarded a treasure.')"
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="If set, save the results as a JSON file in the 'runs' directory."
    )

    parser.add_argument(
        "--use_api",
        action="store_true",
        help="If set, use the OpenAI API to generate results."
    )

    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of times to run the benchmark on a given model."
    )

    args = parser.parse_args()

    print(f"Running benchmark {args.n} times...")
    for _ in range(args.n):

        # Initialize the benchmark with the specified model.
        benchmark = CreativityBenchmark(model_name=args.model, use_api=args.use_api)
        
        # Run the benchmark on the provided prompt.
        results = benchmark.combined_score(args.prompt)
        
        # Generate the results string.
        results_str = print_results(results)
        
        # Print results to console.
        print(results_str)
        
        # Save the results to a file if the --save flag is provided.
        if args.save:
            # Generate a unique run ID using UUID.
            run_id = uuid.uuid4().hex
            
            # Ensure the output directory "runs" exists.
            output_dir = "runs"
            os.makedirs(output_dir, exist_ok=True)
            
            # Sanitize the model name for safe file naming.
            safe_model_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.model)
            filename = f"{safe_model_name}_{run_id}.json"
            output_path = os.path.join(output_dir, filename)

            #add model name to json
            results = { args.model: results }
            
            try:
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=4)
                print(f"\nResults have been saved to '{output_path}'")
            except Exception as e:
                print(f"Error writing to file '{output_path}': {e}")


if __name__ == "__main__":
    main()