
import unittest
from library_analyzer.analyzer import LibraryAnalyzer
from library_analyzer.utils import load_config

class TestLibraryAnalyzerIndexing(unittest.TestCase):
    def setUp(self):
        self.analyzer = LibraryAnalyzer()
        self.config = load_config()

    def test_indexing(self):
        # Charger la configuration
        use_bert = self.config.get("use_bert", True)
        use_whoosh = self.config.get("use_whoosh", True)
        top_k = self.config.get("top_k", 3)

        # Effectuer une analyse et indexer les données
        library_name = "example_library"
        analysis = self.analyzer.analyze_library(library_name)
        text_data = self.analyzer.extract_text_data(analysis)
        self.analyzer.index_data(text_data)

        # Vérifier que les données sont indexées
        search_query = "example search query"
        results = self.analyzer.search(search_query, use_bert=use_bert, use_whoosh=use_whoosh, top_k=top_k)

        # Vérifier que les résultats sont retournés
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        # Vérifier que les résultats contiennent les clés attendues
        for result in results:
            self.assertIn("source", result)
            self.assertIn("text", result)
            self.assertIn("score", result)

if __name__ == "__main__":
    unittest.main()