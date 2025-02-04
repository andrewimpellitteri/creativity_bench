from config import sample_stories, genre_list, edit_requests

class CombinedScoreMixin:
    def combined_score(self, seed_text, weights=None):
        print("\n====== Running Creativity Benchmark Suite =====")
        fa_score, fa_total = self.free_association()
        tel_score = self.telephone_game(seed_text)
        cb_score = self.camels_back(seed_text, edit_requests)
        volume_score = self.dont_repeat_yourself()
        extreme_score = self.extreme_style_transfer(sample_stories, genre_list)
        scores = {
            "free_association": fa_score,
            "telephone_game": tel_score,
            "camels_back": cb_score,
            "diversity": volume_score,
            "extreme_style_transfer": extreme_score
        }
        normalized = {k: v / 100 if k == 'free_association' else v / 10 for k, v in scores.items()}
        weights = weights or {k: 1/len(normalized) for k in normalized}
        composite = sum(normalized[k] * weights[k] for k in normalized)
        return {"scores": scores, "normalized": normalized, "composite": composite}