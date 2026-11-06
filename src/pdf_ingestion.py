
from pathlib import Path
import pymupdf


def inspect_ecg_pdf(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    if path.stat().st_size == 0:
        raise ValueError("PDF file is empty.")

    try:
        document = pymupdf.open(path)
    except Exception as exc:
        raise ValueError(f"Unable to open PDF: {exc}")

    if document.page_count == 0:
        raise ValueError("PDF contains no pages.")

    pages = []
    combined_text = []

    for page_number, page in enumerate(document):
        page_text = page.get_text("text").strip()

        pages.append({
            "page": page_number + 1,
            "width": round(page.rect.width, 2),
            "height": round(page.rect.height, 2),
            "text": page_text,
            "has_images": len(page.get_images()) > 0,
        })

        combined_text.append(page_text)

    full_text = "\n".join(combined_text)

    result = {
        "filename": path.name,
        "page_count": document.page_count,
        "metadata": document.metadata,
        "pages": pages,
        "extracted_text": full_text,
        "has_extractable_text": bool(full_text.strip()),
        "has_embedded_images": any(
            page["has_images"] for page in pages
        ),
    }

    document.close()

    return result
