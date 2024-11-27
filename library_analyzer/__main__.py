import sys
import os
from .utils import parse_json_file, extract_function_signatures, load_config
from .analyzer import LibraryAnalyzer
from .element import ElementType

def analyze_and_display(library_name: str, save_to_file: bool = True):
    analyzer = LibraryAnalyzer()
    print(f"\nAnalyzing library: {library_name}")
    
    analysis = analyzer.analyze_library(library_name)
    
    if save_to_file:
        output_dir = "metrics"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, f"{library_name}_analysis.json")
        analyzer.save_analysis(analysis, output_file)
    
    if 'error' in analysis:
        print(f"\nError during analysis: {analysis['error']}")
    else:
        print("\nAnalysis Summary:")
        print(f"Version: {analysis['metadata']['version']}")
        print(f"Location: {analysis['metadata']['file']}")
        print(f"Number of errors: {len(analysis['metadata']['analysis_errors'])}")
        
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
        print("\nElement counts:")
        for element_type, count in element_counts.items():
            if count > 0:
                print(f"- {element_type}: {count}")

    # Extract text data for indexing
    text_data = analyzer.extract_text_data(analysis)
    # Index the extracted data
    analyzer.index_data(text_data)
    # Perform a sample search
    search_query = "example search query"
    search_results = analyzer.search(search_query)
    print(f"\nSearch results for query '{search_query}':")
    for result in search_results:
        print(f"- Path: {result['path']}, Text: {result['text']}")

    return analysis

def main():
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
            text_data = analyzer.extract_text_data(analysis)
            analyzer.index_data(text_data)
            search_results = analyzer.search(search_query, use_bert=config["use_bert"], use_whoosh=config["use_whoosh"], top_k=config["top_k"])
            print(f"\nSearch results for query '{search_query}':")
            for result in search_results:
                print(f"- Path: {result['path']}, Text: {result['text']}")
        else:
            print("No search query provided, only analysis performed.")
    else:
        print("Please provide a library name as argument")
        print("Example: python -m library_analyzer openai [search_query]")

if __name__ == "__main__": 
    main()
