import pikepdf
from reportlab.pdfgen import canvas
from io import BytesIO

def add_text_to_pdf_with_pikepdf(input_pdf, output_pdf, text, position=(50, 50), font="Helvetica", font_size=12):
    """
    Adds text to an existing PDF while preserving all embedded elements like videos.
    :param input_pdf: Path to the input PDF file.
    :param output_pdf: Path to save the output PDF with added text.
    :param text: Text to add to the PDF.
    :param position: Tuple (x, y) for text position on the page.
    :param font: Font name for the text.
    :param font_size: Font size for the text.
    """
    # Open the existing PDF with pikepdf
    pdf = pikepdf.Pdf.open(input_pdf)
    page_count = len(pdf.pages)

    # Create a new temporary PDF with the text overlay
    packet = BytesIO()
    can = canvas.Canvas(packet)
    can.setFont(font, font_size)
    can.drawString(position[0], position[1], text)
    can.save()
    packet.seek(0)

    # Use pikepdf to append the text overlay to the original PDF
    overlay_pdf = pikepdf.Pdf.open(packet)
    for i in range(page_count):
        page = pdf.pages[i]
        # Merge the text overlay from the new PDF
        page_contents = page.get("/Contents")
        overlay_page_contents = overlay_pdf.pages[0].get("/Contents")
        page.get("/Contents").append(overlay_page_contents)

    # Save the updated PDF
    pdf.save(output_pdf)

    print(f"Text added to {output_pdf} without removing embedded elements.")

# Example usage
if __name__ == "__main__":
    input_pdf = ""  # PDF containing the hidden video
    output_pdf = ""  # Final output PDF
    text = ""  # Text to add
    position = (100, 100)  # Position (x, y) for the text

    add_text_to_pdf_with_pikepdf(input_pdf, output_pdf, text, position)
