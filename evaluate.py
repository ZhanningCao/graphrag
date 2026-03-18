"""
GraphRAG 回答质量评估脚本

评估指标：
1. 关键词召回率 (Key Point Recall) - 标答中的关键要点，模型回答覆盖了多少
2. ROUGE-L - 衡量最长公共子序列的重叠度
3. 人工评分模板 - 输出对比表格方便人工打分

使用方法：
1. 在下方 QA_PAIRS 中填入问题和标准答案的关键要点
2. 运行 run_queries.py 获取模型回答
3. 运行本脚本进行评估
"""

import os
import re
import json

# ============================================================
# 第一步：在这里填入你的问题、标答关键要点、模型回答文件路径
# ============================================================
QA_PAIRS = [
    {
        "question": "MIG/MAG焊接作业有哪些安全注意事项？",
        "answer_file": r"d:\SEU\SRTP项目\graphrag\q2_result.txt",  # 模型回答文件
        # 标答关键要点列表 - 从原始文档中人工提取
        "key_points": [
            "避免接触运动部件",
            "防止导电嘴灼伤",
            "气瓶安全",
            "穿戴防护装备",
            "工作区域通风",
            # 在这里继续添加...
        ],
    },
    {
        "question": "CLOOS免示教焊接系统支持哪些工件类型？",
        "answer_file": r"d:\SEU\SRTP项目\graphrag\q3_result.txt",
        "key_points": [
            # 在这里填入标答的关键要点...
            "请替换为真实的关键要点1",
            "请替换为真实的关键要点2",
        ],
    },
    {
        "question": "焊接机器人日常保养需要检查哪些项目？",
        "answer_file": r"d:\SEU\SRTP项目\graphrag\q4_result.txt",
        "key_points": [
            # 在这里填入标答的关键要点...
            "请替换为真实的关键要点1",
            "请替换为真实的关键要点2",
        ],
    },
]


# ============================================================
# 评估指标计算
# ============================================================

def compute_key_point_recall(answer: str, key_points: list[str]) -> dict:
    """
    关键要点召回率：标答中的每个关键要点，是否在模型回答中被提及。
    这是最直观的评估方式。
    """
    hit = []
    miss = []
    for point in key_points:
        # 把关键要点拆成关键词，检查是否在回答中出现
        keywords = [w for w in point if len(w.strip()) > 0]
        # 简单匹配：关键要点字符串是否出现在回答中
        if point in answer:
            hit.append(point)
        else:
            # 模糊匹配：关键要点中的关键词有多少出现在回答中
            words = re.findall(r'[\u4e00-\u9fff]+', point)  # 提取中文词
            matched = sum(1 for w in words if w in answer)
            if len(words) > 0 and matched / len(words) >= 0.5:
                hit.append(point)
            else:
                miss.append(point)

    recall = len(hit) / len(key_points) if key_points else 0
    return {
        "recall": recall,
        "hit_count": len(hit),
        "total": len(key_points),
        "hit": hit,
        "miss": miss,
    }


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """
    ROUGE-L: 基于最长公共子序列 (LCS) 的 F1 分数。
    适合评估中文文本的整体重叠度。
    """
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)

    m, n = len(ref_chars), len(hyp_chars)
    if m == 0 or n == 0:
        return 0.0

    # 计算 LCS 长度（空间优化）
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr

    lcs_len = prev[n]
    precision = lcs_len / n if n > 0 else 0
    recall = lcs_len / m if m > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    return f1


def evaluate():
    results = []

    for qa in QA_PAIRS:
        question = qa["question"]
        key_points = qa["key_points"]
        answer_file = qa["answer_file"]

        # 读取模型回答
        if os.path.exists(answer_file):
            with open(answer_file, "r", encoding="utf-8") as f:
                model_answer = f.read()
        else:
            model_answer = "(文件不存在，请先运行查询)"

        # 1) 关键要点召回率
        kp_result = compute_key_point_recall(model_answer, key_points)

        # 2) ROUGE-L（用关键要点拼接作为参考文本）
        reference_text = "。".join(key_points)
        rouge_l = compute_rouge_l(reference_text, model_answer)

        result = {
            "question": question,
            "key_point_recall": f"{kp_result['recall']:.1%} ({kp_result['hit_count']}/{kp_result['total']})",
            "rouge_l": f"{rouge_l:.3f}",
            "hit_points": kp_result["hit"],
            "miss_points": kp_result["miss"],
            "answer_length": len(model_answer),
        }
        results.append(result)

    # ---- 输出评估报告 ----
    print("=" * 70)
    print("GraphRAG 回答质量评估报告")
    print("=" * 70)

    for i, r in enumerate(results, 1):
        print(f"\n--- 问题 {i}: {r['question']} ---")
        print(f"  关键要点召回率: {r['key_point_recall']}")
        print(f"  ROUGE-L F1:     {r['rouge_l']}")
        print(f"  回答长度:       {r['answer_length']} 字符")
        if r["hit_points"]:
            print(f"  ✓ 命中的要点:   {r['hit_points']}")
        if r["miss_points"]:
            print(f"  ✗ 遗漏的要点:   {r['miss_points']}")

    # 计算平均分
    avg_recall_values = []
    for qa, r in zip(QA_PAIRS, results):
        kp = compute_key_point_recall(
            open(qa["answer_file"], encoding="utf-8").read() if os.path.exists(qa["answer_file"]) else "",
            qa["key_points"]
        )
        avg_recall_values.append(kp["recall"])

    print(f"\n{'=' * 70}")
    print(f"平均关键要点召回率: {sum(avg_recall_values)/len(avg_recall_values):.1%}")
    print(f"{'=' * 70}")

    # 保存详细报告到文件
    report_path = r"d:\SEU\SRTP项目\graphrag\evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n详细报告已保存到: {report_path}")


if __name__ == "__main__":
    evaluate()
