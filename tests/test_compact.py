import unittest

from src.compression.baselines.compact import CompActCompressor
from src.eval.interfaces import SearchResult


class TestCompActIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        print("\nLoading CompAct model...")
        cls.compressor = CompActCompressor()
        print("✅ CompAct loaded.")

    def test_basic_summary(self):

        query = "What is the capital of France?"

        docs = [
            SearchResult(0,0,"Doc1","Paris is the capital city of France."),
            SearchResult(1,1,"Doc2","France is located in Europe."),
            SearchResult(2,2,"Doc3","Bananas grow in tropical regions.")
        ]

        result = self.compressor.compress(query, docs)

        print("\nCompressed summary:")
        print(result[0].text)

        self.assertEqual(len(result), 1)
        self.assertTrue(len(result[0].text) > 0)


if __name__ == "__main__":
    unittest.main()