"""Run a single GraphRAG query and save result."""
import sys, os
os.chdir(r"d:\SEU\SRTP\graphrag\graphrag\graph_database")
os.environ["PYTHONIOENCODING"] = "utf-8"
from graphrag.cli.query import run_basic_search

query_text = sys.argv[1]
output_file = sys.argv[2]
root = r"d:\SEU\SRTP\graphrag\graphrag\graph_database"

response, _ = run_basic_search(
    data_dir=None, root_dir=root,
    response_type="Multiple Paragraphs",
    streaming=False, query=query_text, verbose=False,
)
with open(output_file, "w", encoding="utf-8") as f:
    f.write(str(response))
print("DONE")
