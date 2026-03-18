"""对比测试：Local Search (无 PCST) vs Local Search + PCST

通过动态修改 settings.yaml 中 use_pcst 来切换模式。
运行前确保 Ollama 已启动且模型可用。
Usage: python test_pcst.py "你的查询问题"
"""
import sys
import os
import re

os.chdir(r"d:\SEU\SRTP项目\graphrag\graph_database")
os.environ["PYTHONIOENCODING"] = "utf-8"

SETTINGS_PATH = r"d:\SEU\SRTP项目\graphrag\graph_database\settings.yaml"
RESULT_DIR = r"d:\SEU\SRTP项目\graphrag"


def set_use_pcst(enabled: bool):
    """修改 settings.yaml 中 use_pcst 的值。"""
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    new_val = "true" if enabled else "false"
    content = re.sub(
        r"(use_pcst:\s*)(true|false)",
        rf"\g<1>{new_val}",
        content,
    )
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [config] use_pcst = {new_val}")


def run_local(query: str, root: str):
    """每次重新导入以获取最新配置。"""
    # 清除已缓存的模块，确保重新读取 settings.yaml
    mods_to_remove = [k for k in sys.modules if "graphrag" in k]
    for m in mods_to_remove:
        del sys.modules[m]

    from graphrag.cli.query import run_local_search
    response, context = run_local_search(
        data_dir=None,
        root_dir=root,
        community_level=0,
        response_type="Multiple Paragraphs",
        streaming=False,
        query=query,
        verbose=False,
    )
    return str(response)


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "MIG/MAG焊接作业有哪些安全注意事项？"
    root = r"d:\SEU\SRTP项目\graphrag\graph_database"

    print("=" * 70)
    print(f"查询: {query}")
    print("=" * 70)

    # 1. Local Search 无 PCST (baseline)
    print("\n>>> [1/2] Local Search (无 PCST, baseline)...")
    set_use_pcst(False)
    try:
        local_plain = run_local(query, root)
        print("Local Search 回答：")
        print(local_plain[:800])
        print("..." if len(local_plain) > 800 else "")
    except Exception as e:
        print(f"Local Search (无 PCST) 失败: {e}")
        local_plain = f"失败: {e}"

    print("\n" + "-" * 70)

    # 2. Local Search + PCST
    print("\n>>> [2/2] Local Search + PCST (子图检索)...")
    set_use_pcst(True)
    try:
        local_pcst = run_local(query, root)
        print("Local Search + PCST 回答：")
        print(local_pcst[:800])
        print("..." if len(local_pcst) > 800 else "")
    except Exception as e:
        print(f"Local Search + PCST 失败: {e}")
        local_pcst = f"失败: {e}"

    # 还原为开启 PCST
    set_use_pcst(True)

    # 保存结果
    with open(os.path.join(RESULT_DIR, "local_plain_result.txt"), "w", encoding="utf-8") as f:
        f.write(f"查询: {query}\n\n")
        f.write(local_plain)
    with open(os.path.join(RESULT_DIR, "local_pcst_result.txt"), "w", encoding="utf-8") as f:
        f.write(f"查询: {query}\n\n")
        f.write(local_pcst)

    print("\n" + "=" * 70)
    print("结果已保存：")
    print(f"  Local Search (无PCST) → local_plain_result.txt")
    print(f"  Local Search + PCST   → local_pcst_result.txt")
    print("=" * 70)


if __name__ == "__main__":
    main()
