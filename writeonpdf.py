import pikepdf
from reportlab.pdfgen import canvas
from io import BytesIO

def add_text_to_pdf_with_pikepdf(input_pdf, output_pdf, text, position=(50, 50), font="Helvetica", font_size=12):

    pdf = pikepdf.Pdf.open(input_pdf)
    page_count = len(pdf.pages)

    packet = BytesIO()
    can = canvas.Canvas(packet)
    can.setFont(font, font_size)
    can.drawString(position[0], position[1], text)
    can.save()
    packet.seek(0)

    overlay_pdf = pikepdf.Pdf.open(packet)
    for i in range(page_count):
        page = pdf.pages[i]
        page_contents = page.get("/Contents")
        overlay_page_contents = overlay_pdf.pages[0].get("/Contents")
        page.get("/Contents").append(overlay_page_contents)

    pdf.save(output_pdf)

    print(f"Text added to {output_pdf} without removing embedded elements.")

if __name__ == "__main__":
    input_pdf = ""  
    output_pdf = "" 
    text = ""  # Text to add
    position = (100, 100)  # Position (x, y) for the text

    add_text_to_pdf_with_pikepdf(input_pdf, output_pdf, text, position)
