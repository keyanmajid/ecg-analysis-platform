
from pathlib import Path
import pymupdf


def extract_pdf_images(file_path, output_dir):
    path = Path(file_path)
    output_dir = Path(output_dir)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    document = pymupdf.open(path)
    extracted = []

    try:
        for page_number, page in enumerate(document):
            for image_number, image_info in enumerate(
                page.get_images(full=True)
            ):
                xref = image_info[0]
                image_data = document.extract_image(xref)

                filename = (
                    f"page_{page_number + 1}_"
                    f"image_{image_number + 1}."
                    f"{image_data['ext']}"
                )

                output_path = output_dir / filename
                output_path.write_bytes(image_data["image"])

                extracted.append({
                    "page": page_number + 1,
                    "image_number": image_number + 1,
                    "width": image_data["width"],
                    "height": image_data["height"],
                    "extension": image_data["ext"],
                    "path": str(output_path),
                    "method": "embedded_image",
                })
    finally:
        document.close()

    return extracted


def render_pdf_pages(file_path, output_dir, dpi=200):
    path = Path(file_path)
    output_dir = Path(output_dir)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if dpi < 72 or dpi > 600:
        raise ValueError("DPI must be between 72 and 600.")

    output_dir.mkdir(parents=True, exist_ok=True)

    document = pymupdf.open(path)
    rendered = []

    try:
        for page_number, page in enumerate(document):
            pixmap = page.get_pixmap(
                dpi=dpi,
                colorspace=pymupdf.csRGB,
                alpha=False
            )

            output_path = output_dir / (
                f"{path.stem}_page_{page_number + 1}_rendered.png"
            )

            pixmap.save(output_path)

            rendered.append({
                "page": page_number + 1,
                "width": pixmap.width,
                "height": pixmap.height,
                "dpi": dpi,
                "path": str(output_path),
                "method": "full_page_render",
            })

    finally:
        document.close()

    return rendered
