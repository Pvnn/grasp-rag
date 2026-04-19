import unittest

from src.compression.baselines.longllmlingua import LongLLMLinguaCompressor
from src.eval.interfaces import SearchResult


class TestLongLLMLinguaIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        print("\nLoading LongLLMLingua...")
        cls.compressor = LongLLMLinguaCompressor()
        print("✅ LongLLMLingua ready.")

    def test_basic_compression(self):

        query = "What is the capital of France?"

        docs = [
            SearchResult(0,0,"Doc1","Paris is the capital of France."),
            SearchResult(1,1,"Doc2","Berlin is the capital of Germany."),
            SearchResult(2,2,"Doc3","Bananas are fruits.")
        ]

        result = self.compressor.compress(query, docs)

        print("\nCompressed prompt:")
        print(result[0].text)

        self.assertEqual(len(result), 1)
        self.assertTrue(len(result[0].text) > 0)


if __name__ == "__main__":
    unittest.main()