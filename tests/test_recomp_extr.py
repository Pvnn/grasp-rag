import unittest

from src.compression.baselines.recomp_extr.compressor import RecompExtractiveCompressor
from src.eval.interfaces import SearchResult


class TestRecompExtractiveIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        print("\nLoading RECOMP Extractive...")
        cls.compressor = RecompExtractiveCompressor()
        print("✅ RECOMP Extractive ready.")

    def test_sentence_selection(self):

        query = "What is the capital of France?"

        docs = [
            SearchResult(0,0,"Doc1","Paris is the capital of France."),
            SearchResult(1,1,"Doc2","Berlin is the capital of Germany."),
            SearchResult(2,2,"Doc3","Bananas are fruits.")
        ]

        result = self.compressor.compress(query, docs)

        print("\nSelected sentences:")
        for r in result:
            print(r.text)

        self.assertTrue(len(result) > 0)


if __name__ == "__main__":
    unittest.main()