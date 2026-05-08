from rag.knowledge.models import (
    BlockType, ChangeType, Workspace, Page, Block, Chunk, DocumentVersion
)

ws = Workspace.new("My Notes")
print("Workspace:", ws.id[:8], ws.name)

pg = Page.new(ws.id, "RAG Overview", "/data/rag.md",
              document_type="markdown", tags=["ai", "rag"])
print("Page:", pg.id[:8], pg.title, "doc_id==id:", pg.doc_id == pg.id)

b1 = Block.new(pg.id, BlockType.HEADING_1, "Retrieval-Augmented Generation", 0)
b2 = Block.new(pg.id, BlockType.PARAGRAPH, "RAG combines retrieval with LLM generation.", 1)
print("Block b1 is_heading:", b1.is_heading)
print("Block b2 is_text_bearing:", b2.is_text_bearing)

ck = Chunk.new(pg.id, b2.content, 0,
               section=b1.content, block_ids=[b1.id, b2.id])
meta = ck.to_chroma_meta(pg, ws)
print("Chunk meta keys:", sorted(meta.keys()))
print("Chunk hash (12):", ck.content_hash[:12])

lc_doc = ck.to_document(pg, ws)
print("LCDocument meta doc_id match:", lc_doc.metadata["doc_id"] == pg.id)

ver = DocumentVersion.new(
    pg.id, 1,
    DocumentVersion.compute_hash([b1, b2]),
    ChangeType.CREATED,
)
print("Version:", ver.version, ver.change_type, ver.content_hash[:12])
print("All OK")
