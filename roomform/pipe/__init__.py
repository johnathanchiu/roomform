"""Pipeline stages. Each stage: run(...) -> a contract document.

scan ─┬─ evidence (completion) ──────────────▶ PatchGraph ─┐
      └─ lifting ─▶ boxes ─▶ reconstruction ─▶ meshes ─────┼─ fuse ─▶ SceneDocument
"""
