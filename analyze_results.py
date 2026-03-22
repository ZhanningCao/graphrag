# -*- coding: utf-8 -*-
"""分析 pcst_answers.jsonl 输出质量"""
import json

lines = open(r'pcst_results/pcst_answers.jsonl', 'r', encoding='utf-8').readlines()
data = [json.loads(l) for l in lines]

unable = [d for d in data if '无法' in d['answer']]
ok_answers = [d for d in data if d not in unable]

print(f"总计: {len(data)} 题")
print(f"回答'无法确定': {len(unable)} 题 ({len(unable)/len(data)*100:.0f}%)")
print(f"有实际回答: {len(ok_answers)} 题")
print()

print("=== 回答'无法确定'的题目 ===")
for d in unable:
    print(f"  {d['item_id']}: prompt_tokens={d['prompt_tokens']}, context_chars={d['context_chars']}")
    print(f"    Q: {d['query'][:70]}")
    print(f"    标答: {d['ground_truth'][:80]}")
    print()

print("=== 有回答的题目对比 ===")
for d in ok_answers:
    gt = d['ground_truth']
    ans = d['answer']
    # 检查标答中的关键词是否出现在回答中
    gt_words = set(gt.replace('。','').replace('，','').replace('、',''))
    ans_words = set(ans.replace('。','').replace('，','').replace('、',''))
    overlap = len(gt_words & ans_words) / max(len(gt_words), 1)
    quality = "GOOD" if overlap > 0.6 else "POOR" if overlap < 0.4 else "FAIR"
    print(f"  {d['item_id']} [{quality}]: overlap={overlap:.0%}")
    print(f"    Q: {d['query'][:70]}")
    print(f"    标答: {gt[:100]}")
    print(f"    回答: {ans[:100]}")
    print()
