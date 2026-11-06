
from pathlib import Path

from src.waveform_region import detect_waveform_regions
from src.pdf_waveform_extractor import render_pdf_pages
from src.pdf_waveform_extractor import extract_pdf_images
from src.pdf_ingestion import inspect_ecg_pdf


def assess_ecg_pdf(pdf_path, output_dir):
    pdf_info = inspect_ecg_pdf(pdf_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered_pages = render_pdf_pages(
        pdf_path,
        output_dir / "rendered",
        dpi=200
    )

    embedded_images = extract_pdf_images(
        pdf_path,
        output_dir / "embedded"
    )

    page_assessments = []

    for page in rendered_pages:
        region_result = detect_waveform_regions(
            page["path"]
        )

        page_assessments.append({
            "page": page["page"],
            "candidate_regions": region_result["regions"],
            "candidate_count": region_result["candidate_count"],
        })

    return {
        "filename": pdf_info["filename"],
        "page_count": pdf_info["page_count"],
        "has_extractable_text": pdf_info["has_extractable_text"],
        "has_embedded_images": pdf_info["has_embedded_images"],
        "embedded_image_count": len(embedded_images),
        "rendered_page_count": len(rendered_pages),
        "page_assessments": page_assessments,
        "digitization_status": "requires_calibration_and_review",
        "clinical_interpretation": False,
    }
