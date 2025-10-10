# MinerU Hierarchy & Section Clustering - Implementation Summary

## 🎯 Overview

Successfully updated the MinerU pipeline to use `_content_list.json` format instead of `_model.json`, which provides **document hierarchy** and **section clustering** metadata.

## ✅ Changes Made

### 1. **CLI Wrapper Update** (`src/Medical_KG_rev/services/mineru/cli_wrapper.py`)

Changed output file from `_model.json` → `_content_list.json`:

```python
# OLD: output_path = output_dir / item.document_id / "vlm" / f"{item.document_id}_model.json"
# NEW: output_path = output_dir / item.document_id / "vlm" / f"{item.document_id}_content_list.json"
```

**Benefits:**

- ✅ Includes `text_level` field (1=H1, 2=H2, 3=H3, etc.)
- ✅ Flatter, cleaner structure
- ✅ Better for section clustering
- ✅ Smaller file size

### 2. **Output Parser Update** (`src/Medical_KG_rev/services/mineru/output_parser.py`)

Added new `_parse_content_list_format()` method:

```python
def _parse_content_list_format(self, content_list: list[dict[str, Any]]) -> ParsedDocument:
    """Parse MinerU _content_list.json format with hierarchy."""
    for block_data in content_list:
        text_level = block_data.get("text_level")  # Heading hierarchy!

        metadata = {}
        if text_level is not None:
            metadata["text_level"] = text_level
            metadata["is_heading"] = True
        # ...
```

**Features:**

- ✅ Preserves `text_level` in block metadata
- ✅ Adds `is_heading` flag for easy filtering
- ✅ Handles images with captions and footnotes
- ✅ Generates UUID-based block IDs

### 3. **Demo Script Update** (`download_and_process_random_papers.py`)

Enhanced block serialization to expose hierarchy:

```python
def serialize_block(block):
    # Promote hierarchy metadata to top level
    if 'text_level' in value:
        block_data['heading_level'] = value['text_level']
    if 'is_heading' in value:
        block_data['is_heading'] = value['is_heading']
```

## 📊 Output Format Comparison

### Before (using `_model.json`)

```json
{
  "blocks": [
    {
      "id": "abc-123",
      "text": "Introduction",
      "page": 0,
      "bbox": [0.1, 0.2, 0.9, 0.3],
      "metadata": {"angle": 0}
    }
  ]
}
```

❌ No hierarchy information
❌ Can't distinguish headings from paragraphs

### After (using `_content_list.json`)

```json
{
  "blocks": [
    {
      "id": "abc-123",
      "text": "Introduction",
      "page": 0,
      "bbox": [0.1, 0.2, 0.9, 0.3],
      "heading_level": 1,
      "is_heading": true,
      "metadata": {
        "text_level": 1,
        "is_heading": true,
        "page": 0,
        "reading_order": null
      }
    },
    {
      "id": "def-456",
      "text": "This is a paragraph under the introduction...",
      "page": 0,
      "bbox": [0.1, 0.35, 0.9, 0.45],
      "metadata": {"page": 0}
    }
  ]
}
```

✅ Clear heading hierarchy (`heading_level: 1`)
✅ Easy to filter headings (`is_heading: true`)
✅ Can group content by sections

## 📈 Real Example Output

**Document:** "Developing and evaluating complex interventions: the new Medical Research Council guidance" (BMJ 2008)

**Statistics:**

- Total blocks: 114
- Total headings: 17
- Level 1 headings: 17
- Pages: 6

**Document Outline:**

```
# RESEARCH METHODS & REPORTING (page 0)
# Developing and evaluating complex interventions: the new Medical Research Council guidance (page 0)
# Revisiting the 2000 MRC framework (page 0)
# What are complex interventions? (page 0)
# Summary points (page 1)
# Box 1 What makes an intervention complex? (page 1)
# Development, evaluation, and implementation (page 1)
# Developing a complex intervention (page 1)
# Box 2 Developing and evaluating complex studies (page 2)
# Assessing feasibility (page 2)
# Evaluating a complex intervention (page 2)
# Assessing effectiveness (page 2)
# Measuring outcomes (page 2)
# Understanding processes (page 2)
# Box 3 Experimental designs for evaluating complex interventions (page 3)
# Box 4 Choosing between randomised and non-randomised designs (page 3)
# Conclusions (page 3)
```

## 🔍 Available Metadata Fields

| Field | Description | Example |
|-------|-------------|---------|
| `text_level` | Heading level (1, 2, 3...) | `1` for H1, `2` for H2 |
| `is_heading` | Boolean flag for headings | `true` / `false` |
| `page_idx` | Page number (0-indexed) | `0`, `1`, `2` |
| `bbox` | Bounding box coordinates | `[x1, y1, x2, y2]` |
| `type` | Block type | `"text"`, `"image"`, `"table"` |
| `text` | Extracted content | Full text string |
| `img_path` | Path to extracted images | For image blocks |
| `image_caption` | Image captions | For image blocks |

## 🎯 Use Cases Enabled

### 1. **Section-Based Chunking**

```python
# Group blocks by section
current_section = None
sections = []

for block in blocks:
    if block.get('is_heading'):
        current_section = {
            'heading': block['text'],
            'level': block['heading_level'],
            'blocks': []
        }
        sections.append(current_section)
    elif current_section:
        current_section['blocks'].append(block)
```

### 2. **Table of Contents Generation**

```python
# Generate TOC
toc = []
for block in blocks:
    if block.get('is_heading'):
        toc.append({
            'level': block['heading_level'],
            'title': block['text'],
            'page': block['page']
        })
```

### 3. **Hierarchical Search**

```python
# Find content under specific heading
def get_section_content(blocks, heading_text):
    in_section = False
    section_blocks = []

    for block in blocks:
        if block.get('is_heading'):
            if heading_text in block['text']:
                in_section = True
                section_blocks.append(block)
            elif in_section:
                break  # Next heading reached
        elif in_section:
            section_blocks.append(block)

    return section_blocks
```

### 4. **Smart Summarization**

```python
# Extract key sections for summary
important_sections = [
    'Introduction',
    'Methods',
    'Results',
    'Conclusions'
]

summaries = {}
for section_name in important_sections:
    content = get_section_content(blocks, section_name)
    summaries[section_name] = content
```

## 🚀 Next Steps

### Potential Enhancements

1. **Multi-level Hierarchy Support**
   - Currently all headings are level 1
   - MinerU may support H2, H3 in some documents
   - Parser is ready for multi-level hierarchy

2. **Section Object Model**
   - Create a `Section` class to represent document hierarchy
   - Nested sections with parent/child relationships
   - Automatic section numbering (1, 1.1, 1.2, etc.)

3. **Semantic Section Types**
   - Classify sections: Abstract, Methods, Results, Discussion
   - Use heading text + position for classification
   - Enable section-specific processing

4. **Cross-Reference Resolution**
   - Link "see Section 2.1" references to actual sections
   - Build citation graphs within documents
   - Track figure/table references

5. **Layout-Based Clustering**
   - Use `bbox` coordinates for column detection
   - Identify multi-column layouts
   - Handle complex page structures (sidebars, callouts)

## 📝 Testing

Verified with real PDFs from OpenAlex:

- ✅ W2497721881: WHO Declaration of Helsinki (5 L1 headings)
- ✅ W2162544110: BMJ paper on complex interventions (17 L1 headings)

Both documents processed successfully with full hierarchy preserved.

## 🔗 Related Files

- `src/Medical_KG_rev/services/mineru/cli_wrapper.py` - CLI integration
- `src/Medical_KG_rev/services/mineru/output_parser.py` - Parsing logic
- `src/Medical_KG_rev/services/mineru/types.py` - Type definitions
- `download_and_process_random_papers.py` - Demo script

## 📚 References

- [MinerU Documentation](https://mineru.readthedocs.io/)
- MinerU output format: `_content_list.json` includes `text_level` for hierarchy
- Alternative outputs: `_model.json` (raw), `.md` (markdown), `_layout.pdf` (annotated)
