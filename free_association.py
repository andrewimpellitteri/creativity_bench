import re
from collections import Counter
from tqdm import tqdm

class FreeAssociationMixin:
    def free_association(self, max_iter=100, estimate_total=True):
        """Test creative word association with unseen species estimation"""
        print("\n=== Free Association Test ===")
        print(f"Model will generate up to {max_iter} words until repetition occurs.")
        
        words = []
        frequencies = Counter()
        pbar = tqdm(total=max_iter, desc="Free Association")
        
        prompt = "Freely associate lists of words or numbers—respond with only one plain word without punctuation or quotes. Just say the next word that comes to mind. Nothing else."
        for i in range(max_iter):
            response = self._generate(prompt).strip().lower()
            # Extract the first token and remove surrounding non-alphanumeric characters
            new_word_raw = response.split()[0] if response else ''
            new_word = re.sub(r'^[^a-z0-9]*', '', new_word_raw)
            new_word = re.sub(r'[^a-z0-9]*$', '', new_word)
            
            if not new_word:
                print(f"Skipping empty word at iteration {i+1}")
                pbar.update(1)
                continue
            
            frequencies[new_word] += 1
            print(f"\nWord {i+1}: {new_word}")
            
            if new_word in words:
                print(f"↳ Repetition detected! '{new_word}' was already used.")
                break
            
            words.append(new_word)
            # Update prompt without quotes around the previous word
            prompt = f"Respond with only one plain word without punctuation or quotes. The next word that comes to mind after {new_word} is:"
            pbar.update(1)
        
        pbar.close()
        print(f"\nGenerated {len(words)} unique words before first repetition.")
        print("Word sequence:", " → ".join(words))
        
        if estimate_total and len(words) > 0:
            N = len(words)
            N1 = sum(1 for f in frequencies.values() if f == 1)
            estimated_total = N + (N1 ** 2) / (2 * N) if N > 0 else 0
            print(f"Estimated total vocabulary size: {estimated_total:.0f}")
            return len(words), estimated_total
        
        return len(words)