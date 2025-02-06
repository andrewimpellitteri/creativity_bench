import random
from tqdm.auto import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class CamelBackMixin:
    def camels_back(self, seed_text, edit_requests, max_edits=10):
        """Test story coherence under diverse modifications using a generated check and VADER sentiment analysis."""
        print("\n=== Camel's Back Test ===")
        print(f"Starting story: {seed_text}")

        story = seed_text
        coherent_edits = 0
        analyzer = SentimentIntensityAnalyzer()

        pbar = tqdm(total=max_edits, desc="Camel's Back")
        for i in range(max_edits):
            print(f"\nEdit Attempt {i + 1}:")

            num_edits = min(random.randint(1, 3), len(edit_requests))
            current_edits = random.sample(edit_requests, num_edits)
            print("Requested edits:")
            for edit in current_edits:
                print(f"- {edit}")

            edit_prompt = (
                f"Modify this story according to these instructions:\n{story}\n\n"
            )
            edit_prompt += "\n".join(f"- {edit}" for edit in current_edits)

            modified_story = self._generate(edit_prompt, temperature=0.8)
            print(f"\nModified story:\n{modified_story}")

            check_prompt = f"""Original story: {story}
                                Modified story: {modified_story}
                                Requested edits:
                                {chr(10).join(f"- {edit}" for edit in current_edits)}

                                Answer these questions with Yes/No and do not respond with anything else:
                                1. Is the modified story still coherent and logical?
                                2. Were all the requested edits applied successfully?
                                3. Is the quality at least as good as the original?"""

            check = self._generate(check_prompt, temperature=0.1)
            print(f"\nQuality check results:\n{check}")

            vs = analyzer.polarity_scores(check)
            compound = vs["compound"]
            print(f"Answer: '{check}' | Compound sentiment: {compound}")

            if compound > 0.5:
                coherent_edits += 1
                pbar.update(1)
            else:
                print("→ Quality check failed! Stopping edits.")
                break

        pbar.close()
        print(f"\nSuccessfully made {coherent_edits} coherent edits.")
        return coherent_edits
