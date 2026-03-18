"""快速 PCST 子图检索单元测试（使用合成数据，不加载真实数据也不调用 LLM）"""
import sys
sys.path.insert(0, r"d:\SEU\SRTP项目\graphrag\graphrag-main\packages\graphrag")

from graphrag.data_model.entity import Entity
from graphrag.data_model.relationship import Relationship
from graphrag.query.context_builder.pcst_subgraph import retrieve_pcst_subgraph


def make_entity(title, rank=1):
    return Entity(id=title, short_id=title, title=title, type="TEST", rank=rank)


def make_rel(src, tgt, weight=1.0, desc=""):
    return Relationship(
        id=f"{src}->{tgt}", short_id=f"{src}->{tgt}",
        source=src, target=tgt, weight=weight, description=desc,
    )


# 构建一个小图:
#   A --5-- B --3-- C
#   |       |       |
#   2       1       4
#   |       |       |
#   D --1-- E --2-- F
#   |
#   1
#   |
#   G --1-- H
#
all_entities = [
    make_entity("A", rank=10),
    make_entity("B", rank=8),
    make_entity("C", rank=6),
    make_entity("D", rank=4),
    make_entity("E", rank=2),
    make_entity("F", rank=5),
    make_entity("G", rank=1),
    make_entity("H", rank=1),
    make_entity("X", rank=1),  # 孤立节点
]

all_relationships = [
    make_rel("A", "B", weight=5.0),
    make_rel("B", "C", weight=3.0),
    make_rel("A", "D", weight=2.0),
    make_rel("B", "E", weight=1.0),
    make_rel("C", "F", weight=4.0),
    make_rel("D", "E", weight=1.0),
    make_rel("E", "F", weight=2.0),
    make_rel("D", "G", weight=1.0),
    make_rel("G", "H", weight=1.0),
]

# 测试1: 选择 A 和 C (不直接相连)，PCST 应该通过 B 把它们连起来
print("=" * 60)
print("测试1: 选择 A 和 C，验证 PCST 找到连通子图")
selected = [all_entities[0], all_entities[2]]  # A, C
pcst_ents, pcst_rels = retrieve_pcst_subgraph(
    selected_entities=selected,
    all_entities=all_entities,
    all_relationships=all_relationships,
    query="test",
    text_embedder=None,
    top_k_nodes=10,
    top_k_edges=5,
    cost_per_edge=0.5,
)
ent_titles = {e.title for e in pcst_ents}
rel_ids = {r.id for r in pcst_rels}
print(f"  选中实体: {sorted(ent_titles)}")
print(f"  选中关系: {sorted(rel_ids)}")
assert "A" in ent_titles, "A 应该在子图中"
assert "C" in ent_titles, "C 应该在子图中"
assert "B" in ent_titles, "B 应该在子图中 (连接 A 和 C)"
print("  ✓ 通过")

# 测试2: 选择单个节点
print("\n" + "=" * 60)
print("测试2: 选择单个节点 A")
selected = [all_entities[0]]  # A only
pcst_ents, pcst_rels = retrieve_pcst_subgraph(
    selected_entities=selected,
    all_entities=all_entities,
    all_relationships=all_relationships,
    query="test",
    text_embedder=None,
    top_k_nodes=10,
    top_k_edges=5,
    cost_per_edge=0.5,
)
ent_titles = {e.title for e in pcst_ents}
print(f"  选中实体: {sorted(ent_titles)}")
print(f"  选中关系: {len(pcst_rels)} 条")
assert "A" in ent_titles, "A 应该在子图中"
print("  ✓ 通过")

# 测试3: 空输入
print("\n" + "=" * 60)
print("测试3: 空输入测试")
pcst_ents, pcst_rels = retrieve_pcst_subgraph(
    selected_entities=[],
    all_entities=all_entities,
    all_relationships=all_relationships,
    query="test",
    text_embedder=None,
)
assert len(pcst_ents) == 0
print("  ✓ 通过")

# 测试4: 选中远端节点 A 和 G (需要通过 D 连接)
print("\n" + "=" * 60)
print("测试4: 选择 A 和 G，验证通过 D 连接")
selected = [all_entities[0], all_entities[6]]  # A, G
pcst_ents, pcst_rels = retrieve_pcst_subgraph(
    selected_entities=selected,
    all_entities=all_entities,
    all_relationships=all_relationships,
    query="test",
    text_embedder=None,
    top_k_nodes=10,
    top_k_edges=5,
    cost_per_edge=0.5,
)
ent_titles = {e.title for e in pcst_ents}
print(f"  选中实体: {sorted(ent_titles)}")
print(f"  选中关系: {len(pcst_rels)} 条")
assert "A" in ent_titles, "A 应该在子图中"
assert "G" in ent_titles, "G 应该在子图中"
assert "D" in ent_titles, "D 应该在子图中 (连接 A 和 G)"
print("  ✓ 通过")

# 测试5: high cost_per_edge 应该产生更小的子图
print("\n" + "=" * 60)
print("测试5: 高 cost_per_edge 应产生更小子图")
selected = [all_entities[0], all_entities[2]]  # A, C
pcst_ents_low, _ = retrieve_pcst_subgraph(
    selected_entities=selected,
    all_entities=all_entities,
    all_relationships=all_relationships,
    query="test",
    text_embedder=None,
    top_k_nodes=10,
    top_k_edges=5,
    cost_per_edge=0.1,
)
pcst_ents_high, _ = retrieve_pcst_subgraph(
    selected_entities=selected,
    all_entities=all_entities,
    all_relationships=all_relationships,
    query="test",
    text_embedder=None,
    top_k_nodes=10,
    top_k_edges=5,
    cost_per_edge=5.0,
)
print(f"  低成本 (0.1): {len(pcst_ents_low)} 实体")
print(f"  高成本 (5.0): {len(pcst_ents_high)} 实体")
assert len(pcst_ents_low) >= len(pcst_ents_high), "高成本应该产生更小或等大的子图"
print("  ✓ 通过")

print("\n" + "=" * 60)
print("所有 PCST 单元测试通过！ ✓✓✓")
