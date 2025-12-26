#Read
#Using PyPDF2 / pypdf
from PyPDF2 import PdfReader

reader = PdfReader("sample.pdf")
for page in reader.pages:
    print(page.extract_text())

#Using pdfminer.six
from pdfminer.high_level import extract_text # for extracting text from PDF

text = extract_text("sample.pdf")
print(text)

#Using PyMuPDF
import fitz  # PyMuPDF # high performance renderer

doc = fitz.open("sample.pdf")
for page in doc:
    print(page.get_text())

#--write
#Using ReportLab
from reportlab.pdfgen import canvas #object to add text,lines,shapes,images etc ina pdf.

c = canvas.Canvas("output.pdf")
c.drawString(100, 750, "Hello, PDF World!")  # x=100, y=750 coordinates.
c.save()   

#Using fpdf2 (lightweight & beginner-friendly)
from fpdf import FPDF#allows to generate pdf files

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=16)
pdf.cell(200, 10, "Hello PDF with fpdf2!", ln=True, align="C")#width,height,in=line break(true= cursor moves to the beginning of the next line), cet the alignment center.
pdf.output("output.pdf")

#--visualizing
#Using pdf2image
from pdf2image import convert_from_path

pages = convert_from_path("sample.pdf", 300)  # 300 DPI for clarity
for i, page in enumerate(pages):#keeping a count of the items
    page.save(f"page_{i+1}.jpg", "JPEG")  # Save as image
    page.show()

#Using PyMuPDF (fitz)
import fitz  # PyMuPDF

doc = fitz.open("sample.pdf")
for i, page in enumerate(doc):
    pix = page.get_pixmap()#render a specific page into a image
    pix.save(f"page_{i+1}.png")
    
#pdf2image → Convert PDF → images (most common).
#PyMuPDF (fitz) → Render high-quality images.