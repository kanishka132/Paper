import fitz  # PyMuPDF
import re

def extract_title_from_pdf(pdf_path):
    try:
        # Open the PDF file
        pdf_document = fitz.open(pdf_path)
        
        # Get the first page
        page = pdf_document[0]
        
        # Extract text from the first page
        text = page.get_text("dict")
        
        # Strategy 1: Look for large, bold text near the top of the page
        title_parts = []
        for block in text["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        if span["size"] > 14 and "bold" in span["font"].lower():
                            line_text += span["text"] + " "
                    if line_text.strip():
                        title_parts.append(line_text.strip())
                        if len(title_parts) == 2:  # Capture up to two lines
                            break
                if len(title_parts) == 2:
                    break
        
        if title_parts:
            title = " ".join(title_parts)
            pdf_document.close()
            return title
        
        # Strategy 2: If not found, look for the first two lines of text that are not all uppercase
        title_parts = []
        for block in text["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    text = " ".join([span["text"] for span in line["spans"]])
                    if text and not text.isupper() and len(text) > 10:
                        title_parts.append(text.strip())
                        if len(title_parts) == 2:
                            break
                if len(title_parts) == 2:
                    break
        
        if title_parts:
            title = " ".join(title_parts)
            pdf_document.close()
            return title
        
        # Strategy 3: If still not found, return the first two lines of text
        first_two_lines = []
        for block in text["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    first_two_lines.append(" ".join([span["text"] for span in line["spans"]]).strip())
                    if len(first_two_lines) == 2:
                        break
            if len(first_two_lines) == 2:
                break
        
        pdf_document.close()
        return " ".join(first_two_lines) if first_two_lines else None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
pdf_path = "A_Survey_on_Machine_Learning_Approaches_and_Its_Techniques.pdf"
title = extract_title_from_pdf(pdf_path)
print("Extracted Title:", title if title else "Title not found.")