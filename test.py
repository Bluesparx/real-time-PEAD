from analyzers.llm_analyzer import LLMAnalyzer

# Initialize the analyzer
analyzer = LLMAnalyzer(mongo_uri="mongodb://localhost:27017/", db_name="bse_data")

result = analyzer.analyze_unanalyzed_pdfs(10)
print("Analysis Result:", result)
