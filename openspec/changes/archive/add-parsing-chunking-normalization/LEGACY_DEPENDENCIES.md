# Legacy Dependency Audit

| Check | Command | Result |
| --- | --- | --- |
| Custom chunkers | `rg "from.*custom_splitters" src/` | No matches |
| Legacy PDF parser | `rg "pdf_parser" src/` | No matches |
| Legacy XML parser | `rg "xml_parser" src/` | No matches |
| Adapter chunk helpers | `rg "\.split_document" src/Medical_KG_rev/adapters/` | No matches |

All adapters now import `chunk_document` from the chunking port. No residual dependencies on the removed legacy modules were detected.
