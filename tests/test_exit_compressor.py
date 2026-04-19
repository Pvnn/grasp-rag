import unittest
import os

from src.compression.exit_baseline import ExitBaselineCompressor
from src.eval.interfaces import SearchResult


class TestExitBaselineIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\nLoading EXIT baseline model...")

        token = os.getenv("HF_TOKEN")

        cls.compressor = ExitBaselineCompressor(
            token=token,
            model_name="doubleyyh/exit-gemma-2b"
        )

        print("✅ EXIT model loaded.")

    def test_basic_compression(self):

        query = "What is the capital of France?"

        docs = [
            SearchResult(0,0,"Doc1","Paris is the capital of France."),
            SearchResult(1,1,"Doc2","Berlin is the capital of Germany."),
            SearchResult(2,2,"Doc3","Bananas are yellow fruits.")
        ]

        result = self.compressor.compress(query, " ".join([d.text for d in docs]))

        print("\nCompressed output:")
        print(result)

        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)


if __name__ == "__main__":
    unittest.main()