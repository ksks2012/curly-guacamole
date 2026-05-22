from rag.knowledge.models import (
    BlockType, ChangeType, Workspace, Page, Block, Chunk, DocumentVersion
)


def test_knowledge_models():
    ws = Workspace.new("My Notes")
    assert ws.name == "My Notes"

    pg = Page.new(ws.id, "RAG Overview", "/data/rag.md",
                  document_type="markdown", tags=["ai", "rag"])
    assert pg.doc_id == pg.id

    b1 = Block.new(pg.id, BlockType.HEADING_1, "Retrieval-Augmented Generation", 0)
    b2 = Block.new(pg.id, BlockType.PARAGRAPH, "RAG combines retrieval with LLM generation.", 1)
    assert b1.is_heading
    assert b2.is_text_bearing

    ck = Chunk.new(pg.id, b2.content, 0,
                   section=b1.content, block_ids=[b1.id, b2.id])
    meta = ck.to_chroma_meta(pg, ws)
    assert "doc_id" in meta

    lc_doc = ck.to_document(pg, ws)
    assert lc_doc.metadata["doc_id"] == pg.id

    ver = DocumentVersion.new(
        pg.id, 1,
        DocumentVersion.compute_hash([b1, b2]),
        ChangeType.CREATED,
    )
    assert ver.version == 1
    assert ver.change_type == ChangeType.CREATED

