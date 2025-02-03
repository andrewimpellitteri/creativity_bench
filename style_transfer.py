import numpy as np
from tqdm.auto import tqdm
import random
import ollama
from scipy.spatial.distance import cosine

class StyleTransferMixin:

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