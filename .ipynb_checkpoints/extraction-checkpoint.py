import fitz  # PyMuPDF
import re

def extract_title_from_pdf(pdf_path):
    """
    Extract the title from the first page of the PDF document.
    
    :param pdf_path: Path to the PDF file.
    :return: Extracted title as a string, or None if not found.
    """
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
        print(f"An error occurred while extracting the title: {e}")
        return None

def extract_abstract_keywords_from_pdf(pdf_path):
    """
    Extract the Abstract and Keywords sections from a PDF document.
    
    :param pdf_path: Path to the PDF file.
    :return: A dictionary containing the extracted abstract and keywords.
    """
    doc = fitz.open(pdf_path)
    abstract_paragraph = None
    keywords = None
    
    abstract_found = False
    keywords_found = False
    
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        page_text = page.get_text("text")
        
        if not abstract_found:
            match_abstract = re.search(r'\babstract\b[^\w]*', page_text, re.IGNORECASE)
            if match_abstract:
                start_index = match_abstract.end()
                abstract_paragraph = page_text[start_index:].strip()
                abstract_paragraph = re.sub(r'\s+', ' ', abstract_paragraph).strip()
                section_start_match = re.search(r'\b(introduction|keywords?|methodology|results|conclusion)\b', abstract_paragraph, re.IGNORECASE)
                if section_start_match:
                    abstract_paragraph = abstract_paragraph[:section_start_match.start()].strip()
                abstract_found = True
        
        if not keywords_found:
            keywords = extract_keywords(page_text)
            if keywords:
                keywords_found = True
        
        if abstract_found and keywords_found:
            break
    
    doc.close()
    return {"Abstract": abstract_paragraph, "Keywords": keywords}

def extract_keywords(text):
    """
    Extract keywords from the text based on the heading 'Keywords' or 'Keyword'.
    
    :param text: Full text extracted from the PDF.
    :return: Extracted keywords or None if not found.
    """
    lines = text.splitlines()
    lines = [line.strip() for line in lines if line.strip()]
    capturing = False
    keywords_content = []

    for line in lines:
        if re.search(r'\b(keywords?|index terms?)\b[\s—:-]*', line, re.IGNORECASE):
            capturing = True
            content_match = re.search(r'\b(keywords?|index terms?)\b[\s—:-]*(.*)', line, re.IGNORECASE)
            if content_match:
                keywords_content.append(content_match.group(2).strip())
            continue
        
        if capturing:
            if '.' in line:
                keywords_content.append(line.split('.')[0].strip())
                break
            keywords_content.append(line)

    return " ".join(keywords_content).strip() if keywords_content else None

def extract_references_from_pdf(pdf_path):
    """
    Extract references or citations from a PDF research paper based on numbering logic.
    
    :param pdf_path: Path to the PDF file.
    :return: A list of extracted references.
    """
    try:
        pdf_document = fitz.open(pdf_path)
        references_text = extract_references_section(pdf_document)
        if not references_text:
            raise ValueError("References section could not be located.")
        references = extract_references_by_numbering(references_text)
        return references
    except Exception as e:
        print(f"An error occurred while extracting references: {e}")
        return []
    finally:
        if 'pdf_document' in locals():
            pdf_document.close()

def extract_references_section(pdf_document):
    references_text = ""
    found_references = False
    continue_next_page = False

    for page_number, page in enumerate(pdf_document):
        text = page.get_text()
        lines = text.splitlines()
        
        for line in lines:
            if not found_references and not continue_next_page:
                # Check if the "References" or "Bibliography" section begins on this line
                if re.search(r'\bReferences\b|\bBibliography\b', line, re.IGNORECASE):
                    found_references = True
            
            # If "References" has been found or we're continuing from the previous page, append lines
            if found_references or continue_next_page:
                references_text += line + "\n"
        
        # Check if we need to continue to the next page
        if found_references and page_number + 1 < len(pdf_document):
            next_page_text = pdf_document[page_number + 1].get_text()
            if not re.search(r'\b(Chapter|Section)\b', next_page_text[:100], re.IGNORECASE):
                continue_next_page = True
            else:
                continue_next_page = False
        else:
            continue_next_page = False

    return references_text.strip() if found_references else None


def extract_references_by_numbering(text):
    lines = text.splitlines()
    references = []
    current_ref = ""
    expected_number = 1

    for line in lines:
        if re.search(r'authorized licensed use', line, re.IGNORECASE):
            continue
        match = re.match(rf'^\(?{expected_number}\)?\.?\s|\[{expected_number}\]\s', line)
        if match:
            if current_ref:
                references.append(current_ref.strip())
            current_ref = line
            expected_number += 1
        else:
            if current_ref:
                current_ref += " " + line
    if current_ref:
        references.append(current_ref.strip())
    return references

# Example usage
if __name__ == "__main__":
    pdf_path = "downloads/1909.03550v1.pdf"
    title = extract_title_from_pdf(pdf_path)
    print("Extracted Title:", title if title else "Title not found.")
    extracted_data = extract_abstract_keywords_from_pdf(pdf_path)
    print("Extracted Abstract:", extracted_data["Abstract"])
    print("Extracted Keywords:", extracted_data["Keywords"])
    references = extract_references_from_pdf(pdf_path)
    print("\nExtracted References:")
    for ref in references:
        print(ref)
