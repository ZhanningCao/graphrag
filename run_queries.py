"""Run multiple GraphRAG queries and save results to files."""
import sys
import os

os.chdir(r"d:\SEU\SRTP项目\graphrag\graph_database")
os.environ["PYTHONIOENCODING"] = "utf-8"

from graphrag.cli.query import run_basic_search

queries = [
    ("MIG/MAG焊接作业有哪些安全注意事项？", "q2_result.txt"),
    ("CLOOS免示教焊接系统支持哪些工件类型？", "q3_result.txt"),
    ("焊接机器人日常保养需要检查哪些项目？", "q4_result.txt"),
]

root = r"d:\SEU\SRTP项目\graphrag\graph_database"

for query_text, output_file in queries:
    print(f"\n{'='*60}")
    print(f"查询: {query_text}")
    print(f"{'='*60}")
    try:
        response, context_data = run_basic_search(
            data_dir=None,
            root_dir=root,
            response_type="Multiple Paragraphs",
            streaming=False,
            query=query_text,
            verbose=False,
        )
        result_path = os.path.join(r"d:\SEU\SRTP项目\graphrag", output_file)
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(f"查询: {query_text}\n\n")
            f.write(str(response))
        print(f"结果已保存到 {output_file}")
    except Exception as e:
        print(f"查询失败: {e}")

print("\n所有查询完成！")
