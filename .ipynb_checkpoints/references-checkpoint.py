import fitz  # PyMuPDF
import re

def extract_references_from_pdf(pdf_path):
    """
    Extract references or citations from a PDF research paper based on numbering logic.
    
    :param pdf_path: Path to the PDF file.
    :return: A list of extracted references.
    """
    try:
        # Open the PDF file
        pdf_document = fitz.open(pdf_path)
        
        # Locate References section and extract text
        references_text = extract_references_section(pdf_document)
        if not references_text:
            raise ValueError("References section could not be located.")

        # Extract references based on numbering
        references = extract_references_by_numbering(references_text)
        return references

    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    finally:
        if 'pdf_document' in locals():
            pdf_document.close()

def extract_references_section(pdf_document):
    """
    Identify and extract the References section from the PDF document.
    
    :param pdf_document: The opened PDF document.
    :return: The References section as a string.
    """
    references_text = ""
    found_references = False
    
    for page in pdf_document:
        text = page.get_text()
        lines = text.splitlines()
        
        for line in lines:
            if not found_references and re.search(r'\bReferences\b|\bBibliography\b', line, re.IGNORECASE):
                found_references = True
            
            if found_references:
                references_text += line + "\n"

    return references_text.strip() if found_references else None

def extract_references_by_numbering(text):
    """
    Extract references based on sequential numbering logic.
    
    :param text: The text of the References section.
    :return: A list of individual references.
    """
    lines = text.splitlines()

    references = []
    current_ref = ""
    expected_number = 1

    for line in lines:
        # Ignore lines containing "Authorized licensed use"
        if re.search(r'authorized licensed use', line, re.IGNORECASE):
            continue

        # Match lines starting with a number (e.g., "1.", "[1]", "(1)")
        match = re.match(rf'^\(?{expected_number}\)?\.?\s|\[{expected_number}\]\s', line)
        if match:
            # Save the current reference if it exists
            if current_ref:
                references.append(current_ref.strip())
            # Start a new reference
            current_ref = line
            expected_number += 1
        else:
            # Append to the current reference if already started
            if current_ref:
                current_ref += " " + line

    # Append the last reference if any
    if current_ref:
        references.append(current_ref.strip())

    return references

# Example usage
if __name__ == "__main__":
    pdf_path = "A_Survey_on_Machine_Learning_Approaches_and_Its_Techniques.pdf"
    references = extract_references_from_pdf(pdf_path)
    print("\nExtracted References:")
    for ref in references:
        print(ref)