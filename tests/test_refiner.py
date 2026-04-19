import unittest

from src.compression.baselines.refiner import RefinerCompressor
from src.eval.interfaces import SearchResult


class TestRefinerIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        print("\nLoading Refiner model...")
        cls.compressor = RefinerCompressor()
        print("✅ Refiner ready.")

    def test_refiner_output(self):

        query = "What is the capital of France?"

        docs = [
            SearchResult(0,0,"Doc1","Paris is the capital city of France."),
            SearchResult(1,1,"Doc2","France is in Europe."),
            SearchResult(2,2,"Doc3","Bananas grow in tropical climates.")
        ]

        result = self.compressor.compress(query, docs)

        print("\nRefined output:")
        print(result[0].text)

        self.assertEqual(len(result), 1)
        self.assertTrue(len(result[0].text) > 0)


if __name__ == "__main__":
    unittest.main()