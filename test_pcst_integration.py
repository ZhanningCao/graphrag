"""集成测试：验证 PCST 子图检索在 Local Search context builder 中正确工作

只测试 context 构建部分，不调用 LLM 生成回答。
"""
import os, sys, asyncio, logging
os.chdir(r"d:\SEU\SRTP项目\graphrag\graph_database")

import pandas as pd
from graphrag.config.load_config import load_config
from graphrag.cli.query import _resolve_output_files
from graphrag.query.factory import get_local_search_engine
from graphrag.query.indexer_adapters import (
    read_indexer_communities,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
    read_indexer_covariates,
)
from graphrag.utils.api import get_embedding_store
from graphrag.config.embeddings import entity_description_embedding


def test_pcst_context():
    root_dir = r"d:\SEU\SRTP项目\graphrag\graph_database"
    config = load_config(root_dir=root_dir)

    print(f"PCST 配置: use_pcst={config.local_search.use_pcst}, "
          f"top_k_nodes={config.local_search.pcst_top_k_nodes}, "
          f"top_k_edges={config.local_search.pcst_top_k_edges}, "
          f"cost_per_edge={config.local_search.pcst_cost_per_edge}")

    # 加载数据
    dataframe_dict = _resolve_output_files(
        config=config,
        output_list=["communities", "community_reports", "text_units",
                      "relationships", "entities"],
        optional_list=["covariates"],
    )

    entities_df = dataframe_dict["entities"]
    communities_df = dataframe_dict["communities"]
    community_reports_df = dataframe_dict["community_reports"]
    text_units_df = dataframe_dict["text_units"]
    relationships_df = dataframe_dict["relationships"]
    covariates_df = dataframe_dict["covariates"]

    community_level = None
    entities_ = read_indexer_entities(entities_df, communities_df, community_level)
    reports_ = read_indexer_reports(community_reports_df, communities_df, community_level)
    relationships_ = read_indexer_relationships(relationships_df)
    text_units_ = read_indexer_text_units(text_units_df)
    covariates_ = read_indexer_covariates(covariates_df) if covariates_df is not None else []

    description_embedding_store = get_embedding_store(
        config=config.vector_store,
        embedding_name=entity_description_embedding,
    )

    print(f"实体: {len(entities_)}, 关系: {len(relationships_)}")

    search_engine = get_local_search_engine(
        config=config,
        reports=reports_,
        text_units=text_units_,
        entities=entities_,
        relationships=relationships_,
        covariates={"claims": covariates_},
        description_embedding_store=description_embedding_store,
        response_type="Multiple Paragraphs",
    )

    context_builder = search_engine.context_builder
    print(f"Context builder 实体数: {len(context_builder.entities)}")
    print(f"Context builder 关系数: {len(context_builder.relationships)}")

    query = "MIG/MAG焊接作业有哪些安全注意事项？"
    print(f"\n查询: {query}")
    print("构建上下文中 (向量检索 + PCST)...")

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    result = context_builder.build_context(
        query=query,
        conversation_history=None,
        **search_engine.context_builder_params,
    )

    context_text = result.context_chunks
    print(f"\n上下文文本长度: {sum(len(c) for c in context_text)} 字符")
    print(f"上下文块数: {len(context_text)}")

    if context_builder._pcst_relationships is not None:
        print(f"\n✅ PCST 子图检索已生效！")
        print(f"  PCST 选中关系数: {len(context_builder._pcst_relationships)}")
        for r in list(context_builder._pcst_relationships)[:10]:
            desc = (r.description or "?")[:40]
            print(f"    {r.source} -> {r.target} [{desc}] (w={r.weight})")
    else:
        print(f"\n❌ PCST 未生效 (_pcst_relationships is None)")


if __name__ == "__main__":
    test_pcst_context()
