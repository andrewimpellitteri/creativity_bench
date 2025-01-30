import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from sentence_transformers import SentenceTransformer
from benchmark import CreativityBenchmark

class TestTelephoneGame(unittest.TestCase):
    def setUp(self):
        self.benchmark = CreativityBenchmark(model_name="deepseek-r1:1.5b")
        # Create a mock embedder that returns consistent embeddings
        self.mock_embedder = MagicMock()
        self.benchmark.embedder = self.mock_embedder
        
    def mock_embeddings(self, text):
        # Return different embeddings based on text to simulate semantic drift
        if "initial" in text.lower():
            return np.array([1.0, 0.0, 0.0])
        elif "modified" in text.lower():
            return np.array([0.8, 0.2, 0.0])
        else:
            return np.array([0.5, 0.5, 0.0])

    def test_convergence_detection(self):
        """Test if the telephone game correctly detects convergence"""
        def mock_generate(prompt, **kwargs):
            if "Expand" in prompt:
                if "initial" in prompt.lower():
                    return "Initial story about a cat."
                elif "modified" in prompt.lower():
                    return "Modified story about a cat."
                else:
                    return "Almost identical story about a cat."
            elif "Summarize" in prompt:
                if "initial" in prompt.lower():
                    return "Modified story about a cat."
                else:
                    return "Almost identical story about a cat."
            return "Initial story about a cat."

        with patch.object(self.benchmark, '_generate', side_effect=mock_generate):
            self.mock_embedder.encode.side_effect = self.mock_embeddings
            iterations = self.benchmark.telephone_game("Initial story about a cat.", max_iter=5)
            self.assertLess(iterations, 5, "Should detect convergence before max iterations")

    def test_early_termination(self):
        """Test if the game terminates when responses become too similar"""
        def mock_generate(prompt, **kwargs):
            return "A cat chased a mouse."  # Always return the same response

        with patch.object(self.benchmark, '_generate', side_effect=mock_generate):
            self.mock_embedder.encode.return_value = np.array([1.0, 0.0, 0.0])
            iterations = self.benchmark.telephone_game("Test story", max_iter=3)
            self.assertEqual(iterations, 1, "Should terminate after first repetition")

    def test_semantic_drift(self):
        """Test if the game continues when stories maintain semantic difference"""
        story_sequence = [
            "A cat chased a mouse.",
            "A feline pursued a rodent.",
            "A predator hunted its prey.",
            "A hunter stalked its target."
        ]
        story_idx = 0
        
        def mock_generate(prompt, **kwargs):
            nonlocal story_idx
            if "Summarize" in prompt:
                story_idx = (story_idx + 1) % len(story_sequence)
            return story_sequence[story_idx]

        with patch.object(self.benchmark, '_generate', side_effect=mock_generate):
            self.mock_embedder.encode.side_effect = lambda x: np.random.rand(3)
            iterations = self.benchmark.telephone_game("Test story", max_iter=4)
            self.assertEqual(iterations, 4, "Should continue when stories remain different")

    def test_timeout_handling(self):
        """Test if the game handles potential timeouts or long responses"""
        def slow_generate(prompt, **kwargs):
            import time
            time.sleep(0.1)  # Simulate slow response
            return "A slow story about a cat."

        with patch.object(self.benchmark, '_generate', side_effect=slow_generate):
            self.mock_embedder.encode.return_value = np.array([1.0, 0.0, 0.0])
            iterations = self.benchmark.telephone_game("Test story", max_iter=3)
            self.assertLessEqual(iterations, 3, "Should handle slow responses gracefully")

    def test_empty_responses(self):
        """Test handling of empty or invalid responses"""
        responses_idx = 0
        
        def mock_generate(prompt, **kwargs):
            nonlocal responses_idx
            responses = ["Initial story.", "", "Final story."]
            response = responses[responses_idx]
            responses_idx = (responses_idx + 1) % len(responses)
            return response

        with patch.object(self.benchmark, '_generate', side_effect=mock_generate):
            self.mock_embedder.encode.return_value = np.array([1.0, 0.0, 0.0])
            with self.assertRaises(ValueError):
                self.benchmark.telephone_game("Test story", max_iter=3)

if __name__ == '__main__':
    unittest.main()