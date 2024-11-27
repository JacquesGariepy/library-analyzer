import sys
import os
import logging
from .utils import parse_json_file, extract_function_signatures, load_config
from .analyzer import LibraryAnalyzer
from .element import ElementType
from .logging_config import setup_logging

# Configurer le logging
setup_logging()
logger = logging.getLogger(__name__)

def analyze_and_display(library_name: str, save_to_file: bool = True):
    """
    Analyze the specified library and display the results.

    Args:
        library_name (str): The name of the library to analyze.
        save_to_file (bool): Whether to save the analysis results to a file.

    Returns:
        dict: The analysis results.
    """
    analyzer = LibraryAnalyzer()
    logging.info(f"\nAnalyzing library: {library_name}")
    
    analysis = analyzer.analyze_library(library_name)
    
    if save_to_file:
        output_dir = "metrics"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, f"{library_name}_analysis.json")
        analyzer.save_analysis(analysis, output_file)
    
    if 'error' in analysis:
        logging.error(f"\nError during analysis: {analysis['error']}")
    else:
        logging.info("\nAnalysis Summary:")
        logging.info(f"Version: {analysis['metadata']['version']}")
        logging.info(f"Location: {analysis['metadata']['file']}")
        logging.info(f"Number of errors: {len(analysis['metadata']['analysis_errors'])}")
        
        def count_elements(data, counts=None):
            if counts is None:
                counts = {t.value: 0 for t in ElementType}
            
            if isinstance(data, dict):
                if 'type' in data and isinstance(data['type'], str):
                    counts[data['type']] = counts.get(data['type'], 0) + 1
                for value in data.values():
                    if isinstance(value, (dict, list)):
                        count_elements(value, counts)
            elif isinstance(data, list):
                for item in data:
                    count_elements(item, counts)
            
            return counts
        
        element_counts = count_elements(analysis)
        logging.info("\nElement counts:")
        for element_type, count in element_counts.items():
            if count > 0:
                logging.info(f"- {element_type}: {count}")

    # Extract text data for indexing
    text_data = analyzer.extract_text_data(analysis)
    # Index the extracted data
    analyzer.index_data(text_data)
    # Charger la configuration
    config = load_config()
    use_bert = config.get("use_bert", True)
    use_whoosh = config.get("use_whoosh", True)
    top_k = config.get("top_k", 3)

    # Perform a sample search
    search_query = "example search query"
    search_results = analyzer.search(search_query, use_bert=use_bert, use_whoosh=use_whoosh, top_k=top_k)
    logging.info(f"\nSearch results for query '{search_query}':")
    for result in search_results:
        path = result.get('path', 'N/A')
        logging.info(f"- Path: {path}, Text: {result['text']}")

    return analysis

def perform_search(analyzer, analysis, search_query):
    """
    Perform a search on the analyzed data.

    Args:
        analyzer (LibraryAnalyzer): The analyzer instance.
        analysis (dict): The analysis results.
        search_query (str): The search query.

    Returns:
        list: The search results.
    """
    text_data = analyzer.extract_text_data(analysis)
    analyzer.index_data(text_data)
    # Charger la configuration
    config = load_config()
    use_bert = config.get("use_bert", True)
    use_whoosh = config.get("use_whoosh", True)
    top_k = config.get("top_k", 3)

    search_results = analyzer.search(search_query, use_bert=use_bert, use_whoosh=use_whoosh, top_k=top_k)
    return search_results

def main():
    """
    Main entry point for the script.
    """
    if len(sys.argv) > 1:
        library_name = sys.argv[1]
        config = load_config()
        analyzer = LibraryAnalyzer()
        
        # Analyze the specified library
        analysis = analyzer.analyze_library(library_name)
        
        # Save analysis to JSON file
        output_dir = "metrics"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, f"{library_name}_analysis.json")
        analyzer.save_analysis(analysis, output_file)
        
        # Optionally perform semantic search
        if len(sys.argv) > 2:
            search_query = sys.argv[2]
            search_results = perform_search(analyzer, analysis, search_query)
            logging.info(f"\nSearch results for query '{search_query}':")
            for result in search_results:
                path = result.get('path', 'N/A')
                logging.info(f"- Path: {path}, Text: {result['text']}")
        else:
            logging.info("No search query provided, only analysis performed.")
    else:
        logging.error("Please provide a library name as argument")
        logging.info("Example: python -m library_analyzer openai [search_query]")

if __name__ == "__main__":
    main()
