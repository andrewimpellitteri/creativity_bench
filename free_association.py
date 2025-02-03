from tqdm.auto import tqdm
from collections import Counter

class FreeAssociationMixin:
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