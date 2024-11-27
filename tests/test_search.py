import unittest
from unittest.mock import MagicMock
from library_analyzer import perform_search, LibraryAnalyzer
from library_analyzer.analyzer import LibraryAnalyzer
from library_analyzer.utils import load_config

class TestLibraryAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = LibraryAnalyzer()
        self.config = load_config()

    def test_search_with_config(self):
        # Charger la configuration
        use_bert = self.config.get("use_bert", True)
        use_whoosh = self.config.get("use_whoosh", True)
        top_k = self.config.get("top_k", 3)

        # Effectuer une recherche
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

class TestPerformSearch(unittest.TestCase):

    def setUp(self):
        self.analyzer = MagicMock(spec=LibraryAnalyzer)
        self.analysis = {
            'data': [
                {'text': 'example search query'},
                {'text': 'another example'},
                {'text': 'search term'}
            ]
        }

    def test_search_single_term(self):
        search_query = 'example'
        results = perform_search(self.analyzer, self.analysis, search_query)
        self.assertEqual(len(results), 2)
        self.assertIn('example search query', results)
        self.assertIn('another example', results)

    def test_search_multiple_terms(self):
        search_query = 'example search'
        results = perform_search(self.analyzer, self.analysis, search_query)
        self.assertEqual(len(results), 1)
        self.assertIn('example search query', results)

    def test_search_no_results(self):
        search_query = 'nonexistent'
        results = perform_search(self.analyzer, self.analysis, search_query)
        self.assertEqual(len(results), 0)

    def test_search_empty_query(self):
        search_query = ''
        results = perform_search(self.analyzer, self.analysis, search_query)
        self.assertEqual(len(results), 0)

if __name__ == "__main__":
    unittest.main()