# IIIF-OCR-CLI

## Installation

Install using uv:

```bash
git clone https://github.com/aguilarm-umd/iiif_ocr.git
cd iiif_ocr
uv sync
```

## Usage

```bash
Usage: iiif_ocr [OPTIONS] MANIFEST

  Generate an hOCR file from a IIIF Manifest

Options:
  --visualize  Visualize Bounding Boxes
  --gpu        Use GPU when using OCR engine
  --help       Show this message and exit.
```

`MANIFEST` - URL or file path to the IIIF manifest JSON file

### Options

* `--visualize` - Output additional image showing bounding boxes
* `--gpu` - Use GPU acceleration for OCR processing
**Note: The GPU flag doesn't do anything as of now**

## Output

The tool creates a `downloads/` directory organized by resource ID:

``` text
downloads/
  <UUID-of-Resource-ID/
    page_0_bboxes.jpeg # Outputted with --visualize flag
    page_0_layout_visualization.jpeg # Outputted with --visualize flag
    page_0_ocr_visualization.jpeg # Outputted with --visualize flag
    page_0.html
    page_0.jpeg
    ...
```
