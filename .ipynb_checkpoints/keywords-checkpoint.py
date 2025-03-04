import fitz  # PyMuPDF
import re

def extract_keywords_from_pdf(pdf_path):
    """
    Extract the Keywords section from a PDF document.
    
    :param pdf_path: Path to the PDF file.
    :return: Extracted keywords or None if not found.
    """
    try:
        # Open the PDF file
        pdf_document = fitz.open(pdf_path)
        text = ""

        # Extract text from all pages
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += page.get_text()

        # Close the PDF document
        pdf_document.close()

        # Extract keywords
        keywords = extract_keywords(text)
        return keywords

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def extract_keywords(text):
    """
    Extract keywords from the text based on the heading 'Keywords' or 'Keyword'.
    
    :param text: Full text extracted from the PDF.
    :return: Extracted keywords or None if not found.
    """
    # Normalize and split the text into lines
    lines = text.splitlines()
    lines = [line.strip() for line in lines if line.strip()]  # Remove empty lines

    capturing = False
    keywords_content = []

    for line in lines:
        # Check for 'Keywords' or 'Keyword' heading followed by a symbol or space
        if re.search(r'\b(keywords?|index terms?)\b[\s—:-]*', line, re.IGNORECASE):
            capturing = True
            
            # Immediately capture content after the heading
            content_match = re.search(r'\b(keywords?|index terms?)\b[\s—:-]*(.*)', line, re.IGNORECASE)
            if content_match:
                # Capture everything after the heading up to a full stop
                keywords_content.append(content_match.group(2).strip())
            continue
        
        if capturing:
            # Stop capturing when a full stop is encountered
            if '.' in line:
                # Capture everything up to the first full stop and stop capturing
                keywords_content.append(line.split('.')[0].strip())
                break
            
            # Append line to keywords content if no full stop is found yet
            keywords_content.append(line)

    return " ".join(keywords_content).strip() if keywords_content else None

# Example usage
if __name__ == "__main__":
    pdf_path = "A_Survey_on_Machine_Learning_Approaches_and_Its_Techniques.pdf"  # Replace with your PDF file path
    keywords = extract_keywords_from_pdf(pdf_path)

    print("\nKeywords:")
    print(keywords if keywords else "Keywords not found.")