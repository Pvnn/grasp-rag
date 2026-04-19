import unittest

from src.compression.baselines.recomp_abst import RECOMPAbstractiveCompressor
from src.eval.interfaces import SearchResult


class TestRecompAbstractiveIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        print("\nLoading RECOMP Abstractive...")
        cls.compressor = RECOMPAbstractiveCompressor()
        print("✅ RECOMP Abstractive ready.")

    def test_summary_generation(self):

        query = "What is the capital of France?"

        docs = [
            SearchResult(0,0,"Doc1","Paris is the capital city of France."),
            SearchResult(1,1,"Doc2","France is located in Europe."),
            SearchResult(2,2,"Doc3","Bananas are fruits.")
        ]

        result = self.compressor.compress(query, docs)

        print("\nGenerated summary:")
        print(result[0].text)

        self.assertEqual(len(result), 1)
        self.assertTrue(len(result[0].text) > 0)


if __name__ == "__main__":
    unittest.main()