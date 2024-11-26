import yaml
from library_analyzer import LibraryAnalyzer

def load_config(file="config.yaml"):
    with open(file, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    analyzer = LibraryAnalyzer()
    
    # Analyze a sample library
    library_name = "mistralai"
    analysis = analyzer.analyze_library(library_name)
    
    # Extract text data and index it
    text_data = analyzer.extract_text_data(analysis)
    analyzer.index_data(text_data)
    
    # Perform a semantic search and print results
    search_query = "example search query"
    search_results = analyzer.search(search_query, use_bert=config["use_bert"], use_whoosh=config["use_whoosh"], top_k=config["top_k"])
    print(f"\nSearch results for query '{search_query}':")
    for result in search_results:
        print(f"- Path: {result['path']}, Text: {result['text']}")

if __name__ == "__main__":
    main()