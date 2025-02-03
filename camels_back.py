import random
from tqdm.auto import tqdm

class CamelBackMixin:
    edit_requests = [
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

            check = check.lower()
            
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