---
name: docx-report-editorial
description: |
  A native .docx styled as an editorial report: "Word report", "Word 文档",
  "memo", "proposal", "研究报告". Opens cleanly in Word / Google Docs /
  Pages with real heading styles and a cover page.
when_to_use: |
  A polished .docx meant to be read as a document and further edited in
  Word. Use pdf-report-editorial when the deliverable is a fixed printable
  PDF, and pptx-editorial when the user wants slides.
tags:
  - docx
  - word
  - report
  - document
  - editorial
---

# Editorial Report (.docx)

You will generate one `.docx` file via python-docx by writing a Python
program and running it through the `execute_python_code` tool. Save to
the workspace, then report the path + a 1-line content summary.

## 📦 Required runtime packages

The sandboxed `execute_python_code` ships with `pandas`, `numpy`,
`matplotlib`, `openpyxl`, and **`python-docx>=1.1.0`** preinstalled — no
extra installation step is needed. The import name is **`docx`**, not
`python_docx`:

```python
from docx import Document          # ✅ correct
import python_docx                 # ❌ ModuleNotFoundError
```

> **Note:** `execute_python_code` accepts only `code` and
> `capture_output` arguments; there is no `packages` parameter.
> All required libraries are already available in the sandbox image.

## 💾 How to save the file

`execute_python_code` already runs with the task's output directory as
the current working directory. Save the document with a **plain
filename** (no path), e.g.:

```python
doc.save("market_expansion_report.docx")   # ✅ correct
```

Do NOT use:
- `doc.save("/workspace/foo.docx")` — `/workspace` does not exist on this host
- `doc.save("output/foo.docx")` — there is no nested `output/` subdir
- BytesIO + base64 round-trips — just save to disk directly.

Verify the content, not the byte count: an empty `Document()` already saves
at ~36 KB, so a size threshold cannot tell a real report from a blank one.
Re-open the file and count what you wrote:

```python
from docx import Document

check = Document("market_expansion_report.docx")
assert len(check.paragraphs) > 10 and len(check.tables) >= 1
```

### 🔗 Make it clickable in chat — REQUIRED

The Python executor returns a `markdown_link` field in its response for
every workspace file it generated (or a `file_refs[]` array with one
entry per file). **Read the tool's response** and use the returned
`markdown_link` string verbatim. In your final answer, the **first line
MUST be that chip link itself** — bare markdown, NOT inside backticks,
NOT presented as "file_id: UUID":

✅ **CORRECT** (renders as clickable chip — chat UI looks for this exact pattern):

    [market_expansion_report.docx](file:20fae785-3823-4906-b385-d0e8a7807dc8)

The UUID comes from the executor response. Do not fabricate one.

❌ **WRONG — common failure that renders as plain text**:

    已生成报告:
    - 文件名: `market_expansion_report.docx`
    - file_id: `20fae785-3823-4906-b385-d0e8a7807dc8`

❌ **ALSO WRONG — chip link inside a code fence**: ` ```[name](file:UUID)``` `
   suppresses markdown so the link won't render as a chip.

❌ **ALSO WRONG — calling `get_file_info(...)` to "fetch" the file_id**.
   Its `FileInfo` return shape does not include `file_id`; the chip
   reference is already on the executor result.

## ⚠️ Hard rules — NO exceptions

0. **MATCH THE USER'S LANGUAGE.** If the prompt is Chinese (中文), ALL
   document text (cover kicker, headings, body, table headers, captions,
   footer) must be in Chinese. Translate template phrases like
   `EXECUTIVE SUMMARY` → `摘要`, `FINDINGS` → `调查结果`,
   `RECOMMENDATIONS` → `建议`, `As of YYYY-MM-DD` → `截至 YYYY-MM-DD`.
   Never leave English kickers in a Chinese report.
1. **One palette only.** Pick one of the 5 palettes below; use only its
   **4 hex values** (`ink`, `paper`, `paper_tint`, `ink_tint`). Define a
   `palette = {...}` dict ONCE at the top of the script and reference
   `palette["ink"]` etc. everywhere — do not copy literal hex values into
   individual styling calls.
2. **Two fonts only.** Headings = `Georgia` (serif, present on all OSes).
   Body = `Calibri` (sans, Word default). No custom fonts — recipients
   won't have them and Word falls back to Times.
   **For Chinese documents**: both render Chinese via system fallback
   (PingFang on macOS, Microsoft YaHei on Windows) — do not switch fonts.
3. **Use real Word styles, not manual formatting.** Headings must use the
   built-in `Heading 1` / `Heading 2` / `Heading 3` styles so Word's
   navigation pane and auto table-of-contents work. A document where every
   heading is just bold 18pt body text is broken — it has no outline.
4. **Forbidden:**
   - WordArt, drop shadows, glow, 3-D effects, gradient fills
   - clipart, emoji as decoration, stock-photo placeholders
   - centered body paragraphs (left-align / justify only)
   - all-caps body text (kickers and labels only)
   - colored hyperlinks other than `ink` (links = ink + underline)
   - more than one heading font
   - Comic Sans, Arial Black, Times New Roman as a deliberate choice
5. **Real content only.** No lorem ipsum, no `[Title here]` placeholders,
   no fabricated statistics, no fake citations. If a section has no user
   data, drop the section rather than padding it.
6. **Failure honesty — NEVER fake the deliverable.**
   - If `execute_python_code` raises after multiple retries, STOP and report
     the actual error. Do not write a stub file like
     `write_file("report.docx", "placeholder")` to make the chip appear.
   - The final answer must reflect what was actually written. Do not
     describe sections or tables that aren't in the saved `.docx`.

## 🎨 Palettes — pick ONE

Same palettes as `pdf-report-editorial` and `pptx-editorial` — keep the
editorial family visually consistent. Each: `ink` (body text + rules),
`paper` (page / cell background), `paper_tint` (table band + callout bg),
`ink_tint` (kickers, captions, footer). The keys are underscored — the dict
below is what every snippet indexes into.

- **Monocle** (default / business / tech / policy)
  ink `0A0A0B` · paper `F1EFEA` · paper_tint `E8E5DE` · ink_tint `18181A`
- **Indigo Porcelain** (research / data-heavy)
  ink `0A1F3D` · paper `F1F3F5` · paper_tint `E4E8EC` · ink_tint `152A4A`
- **Forest Ink** (sustainability / impact)
  ink `1A2E1F` · paper `F5F1E8` · paper_tint `ECE7DA` · ink_tint `253D2C`
- **Kraft Paper** (humanities / qualitative)
  ink `2A1E13` · paper `EEDFC7` · paper_tint `E0D0B6` · ink_tint `3A2A1D`
- **Dune** (art / design / fashion criticism)
  ink `1F1A14` · paper `F0E6D2` · paper_tint `E3D7BF` · ink_tint `2D2620`

Define it once at the top of the script, then index it everywhere:

```python
palette = {"ink": "0A0A0B", "paper": "F1EFEA",
           "paper_tint": "E8E5DE", "ink_tint": "18181A"}   # Monocle
```

python-docx takes hex without the `#` prefix, via
`RGBColor.from_string(palette["ink"])`.

## ✒️ Typography (python-docx Pt sizes)

| Role | Style | Family | Size | Weight |
|---|---|---|---|---|
| Cover title | `Title` | Georgia | 40pt | regular |
| Cover subtitle / dek | body run | Calibri | 14pt | italic |
| Kicker (small caps label) | body run | Calibri | 9pt, uppercase | bold, ink_tint |
| H1 section | `Heading 1` | Georgia | 22pt | regular |
| H2 subsection | `Heading 2` | Georgia | 16pt | regular |
| H3 sub-subsection | `Heading 3` | Calibri | 12pt | bold |
| Body paragraph | `Normal` | Calibri | 11pt | regular, line 1.4 |
| Pull quote | `Intense Quote` | Georgia | 14pt | italic |
| Table header | table run | Calibri | 10pt | bold, paper on ink |
| Table body | table run | Calibri | 10pt | regular |
| Caption / footnote | `Caption` | Calibri | 9pt | italic, ink_tint |

## 📐 Page setup

Set margins, size and orientation on each `section` before adding content.

```python
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.section import WD_ORIENT

doc = Document()
sec = doc.sections[0]

# A4 portrait with editorial margins
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.orientation = WD_ORIENT.PORTRAIT
sec.top_margin, sec.bottom_margin = Cm(2.5), Cm(2.5)
sec.left_margin, sec.right_margin = Cm(2.2), Cm(2.2)
```

⚠️ **Orientation gotcha:** setting `sec.orientation = WD_ORIENT.LANDSCAPE`
does NOT swap the page dimensions — Word reads the width/height, not the
flag. You must swap them yourself:

```python
sec.orientation = WD_ORIENT.LANDSCAPE
sec.page_width, sec.page_height = Cm(29.7), Cm(21.0)   # swap explicitly
```

Set the default body font once on the `Normal` style so every paragraph
inherits it (including CJK, which needs the `eastAsia` attribute):

```python
from docx.oxml.ns import qn

normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(11)
normal.font.color.rgb = RGBColor.from_string(palette["ink"])
rpr = normal.element.get_or_add_rPr()
rpr.get_or_add_rFonts().set(qn("w:eastAsia"), "Microsoft YaHei")
normal.paragraph_format.line_spacing = 1.4
normal.paragraph_format.space_after = Pt(8)
```

## 🏛️ Cover page pattern

The cover is its own section so the body can restart page numbering and
use different headers. Structure: kicker → title → dek → rule → meta row.

```python
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK

def add_kicker(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text.upper())
    run.font.name, run.font.size, run.font.bold = "Calibri", Pt(9), True
    run.font.color.rgb = RGBColor.from_string(palette["ink_tint"])
    return p

# --- cover ---
doc.add_paragraph().paragraph_format.space_after = Pt(160)  # vertical push
add_kicker(doc, "Research Brief")

title = doc.add_paragraph(style="Title")
title_run = title.add_run("Agent Infrastructure in 2026")
title_run.font.name, title_run.font.size = "Georgia", Pt(40)
title_run.font.color.rgb = RGBColor.from_string(palette["ink"])

dek = doc.add_paragraph()
dek_run = dek.add_run("Market shape, adoption curves, and the shift to "
                      "production-grade systems.")
dek_run.font.name, dek_run.font.size, dek_run.font.italic = "Calibri", Pt(14), True

meta = doc.add_paragraph()
meta_run = meta.add_run("Xagent Team · 2026-05-14")   # middle dot, not hyphen
meta_run.font.size = Pt(10)
meta_run.font.color.rgb = RGBColor.from_string(palette["ink_tint"])

# A new section — not just a page break — is what lets the body restart page
# numbering and carry its own header/footer.
from docx.enum.section import WD_SECTION

body = doc.add_section(WD_SECTION.NEW_PAGE)
```

## 🔠 Heading hierarchy

Always use the built-in styles, then restyle the style object once — not
each heading individually:

```python
for name, family, size in (("Heading 1", "Georgia", 22),
                           ("Heading 2", "Georgia", 16),
                           ("Heading 3", "Calibri", 12)):
    st = doc.styles[name]
    st.font.name, st.font.size = family, Pt(size)
    st.font.color.rgb = RGBColor.from_string(palette["ink"])
    st.font.bold = (name == "Heading 3")
    st.paragraph_format.space_before = Pt(18)
    st.paragraph_format.space_after = Pt(6)

doc.add_heading("Executive Summary", level=1)
doc.add_paragraph("...")
doc.add_heading("Market Size", level=2)
```

⚠️ `doc.add_heading(..., level=0)` applies the `Title` style, not a
heading — use it only on the cover. Body sections start at `level=1`.

Pull quotes use the built-in style too, so they stay in the outline-free
body flow:

```python
quote = doc.add_paragraph(style="Intense Quote")
quote_run = quote.add_run("Adoption is no longer the constraint; "
                          "operating cost is.")
quote_run.font.name, quote_run.font.size = "Georgia", Pt(14)
quote_run.font.italic = True
```

## 📊 Table styling

python-docx has no API for cell shading or borders, so both need raw
OOXML. These two helpers are the whole toolkit — copy them verbatim.

```python
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# Both w:tcPr and w:tcBorders are ordered sequences: Word rejects the part when
# children appear out of order, and each tag may appear at most once. Appending
# is wrong on both counts, so everything below inserts at the schema position.
_TCPR_ORDER = ("cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders",
               "shd", "noWrap", "tcMar", "textDirection", "tcFitText",
               "vAlign", "hideMark")
_BORDER_ORDER = ("top", "left", "bottom", "right", "insideH", "insideV",
                 "tl2br", "tr2bl")


def _put(parent, tag, order):
    """Replace parent's <w:{tag}> child, keeping the schema's element order."""
    for stale in parent.findall(qn(f"w:{tag}")):
        parent.remove(stale)
    el = OxmlElement(f"w:{tag}")
    rank = order.index(tag)
    later = [child for child in parent
             if child.tag.split("}")[1] in order
             and order.index(child.tag.split("}")[1]) > rank]
    if later:
        later[0].addprevious(el)
    else:
        parent.append(el)
    return el


def shade_cell(cell, hex_fill):
    """Solid background fill for one table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = _put(tcPr, "shd", _TCPR_ORDER)
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)

def set_row_border(row, edge, hex_color, sz=8):
    """Hairline on one edge of every cell in a row, e.g. 'top' or 'bottom'.
    sz is in 1/8 pt. Re-styling the same edge replaces it."""
    for cell in row.cells:
        tcPr = cell._tc.get_or_add_tcPr()
        borders = tcPr.find(qn("w:tcBorders"))
        if borders is None:
            borders = _put(tcPr, "tcBorders", _TCPR_ORDER)
        el = _put(borders, edge, _BORDER_ORDER)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(sz))          # 8 = 1pt
        el.set(qn("w:color"), hex_color)


def clear_cell_border(cell, edge):
    """Remove one edge, e.g. the verticals a full-grid table style draws."""
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = _put(tcPr, "tcBorders", _TCPR_ORDER)
    _put(borders, edge, _BORDER_ORDER).set(qn("w:val"), "nil")
```

Editorial table rules — **horizontal rules only, no vertical borders**:

```python
from docx.enum.table import WD_TABLE_ALIGNMENT

rows = [("Region", "Revenue", "YoY"),
        ("North America", "4.2M", "+18%"),
        ("EMEA", "2.8M", "+11%")]

table = doc.add_table(rows=len(rows), cols=3)
table.style = "Table Grid"          # verticals stripped per cell below
table.alignment = WD_TABLE_ALIGNMENT.CENTER
table.autofit = True

for r, record in enumerate(rows):
    for c, value in enumerate(record):
        cell = table.cell(r, c)
        cell.text = str(value)
        para = cell.paragraphs[0]
        # numbers right-aligned, text left-aligned
        para.alignment = (WD_ALIGN_PARAGRAPH.RIGHT if c > 0 and r > 0
                          else WD_ALIGN_PARAGRAPH.LEFT)
        run = para.runs[0]
        run.font.name, run.font.size = "Calibri", Pt(10)
        if r == 0:                                   # header row
            run.font.bold = True
            run.font.color.rgb = RGBColor.from_string(palette["paper"])
            shade_cell(cell, palette["ink"])         # ink header band
        elif r % 2 == 0:                             # zebra band
            shade_cell(cell, palette["paper_tint"])
            run.font.color.rgb = RGBColor.from_string(palette["ink"])
        else:
            run.font.color.rgb = RGBColor.from_string(palette["ink"])

# "Table Grid" draws a full grid; nil out the verticals to keep it editorial.
for table_row in table.rows:
    for table_cell in table_row.cells:
        for vertical in ("left", "right", "insideV"):
            clear_cell_border(table_cell, vertical)

set_row_border(table.rows[0], "bottom", palette["ink"])
set_row_border(table.rows[-1], "bottom", palette["ink"])

caption = doc.add_paragraph(style="Caption")
caption_run = caption.add_run("Table 1 — Revenue by region, FY2026.")
caption_run.font.size, caption_run.font.italic = Pt(9), True
caption_run.font.color.rgb = RGBColor.from_string(palette["ink_tint"])
```

Repeat the header row across page breaks for tables longer than a page:

```python
# w:trPr allows at most one w:tblHeader, so replace rather than append -- a
# second one is schema-invalid, and a loop over tables would add one per pass.
trPr = table.rows[0]._tr.get_or_add_trPr()
for stale in trPr.findall(qn("w:tblHeader")):
    trPr.remove(stale)
trPr.append(OxmlElement("w:tblHeader"))
```

## 📝 Output checklist

- [ ] `from docx import Document` (import name is `docx`, not `python_docx`)
- [ ] LANGUAGE matches the user's prompt — no English kickers in a ZH report
- [ ] One palette, only its 4 hex values appear anywhere
- [ ] Only Georgia + Calibri; `Normal` style carries the body font + eastAsia
- [ ] Real `Heading 1/2/3` styles used (Word navigation pane shows an outline)
- [ ] Page size, margins, and orientation set explicitly on the section
      (landscape = swap width/height, not just the orientation flag)
- [ ] Cover page present, followed by an explicit page break
- [ ] Tables: ink header band, `paper_tint` zebra rows, horizontal rules only,
      numbers right-aligned, caption below
- [ ] No fabricated data, no lorem ipsum, no placeholder headings
- [ ] File saved with a plain filename, then re-opened and its paragraph /
      table counts asserted (byte size proves nothing — empty is ~36 KB)
- [ ] **Final answer FIRST LINE is `[filename](file:UUID)`** as bare markdown

Then write the .docx and report path + which palette + which sections.
