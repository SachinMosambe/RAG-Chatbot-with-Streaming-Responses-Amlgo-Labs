import os
from PyPDF2 import PdfReader
import re
from tqdm import tqdm

def clean_text(text):
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove headers and footers (assuming they're on separate lines)
    text = re.sub(r'^\s*Page \d+.*$', '', text, flags=re.MULTILINE)
    
    # Remove HTML tags if any
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove special characters and normalize spaces
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    text = text.strip()
    
    return text

def process_pdfs(input_folder, output_file):
    # Create data folder if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Get all PDF files from the input folder
    pdf_files = [f for f in os.listdir(input_folder) if f.endswith('.pdf')]
    
    all_text = []
    
    # Process each PDF file
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(input_folder, pdf_file)
        
        try:
            # Read PDF file
            reader = PdfReader(pdf_path)
            
            # Extract text from each page
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    # Clean the extracted text
                    cleaned_text = clean_text(text)
                    if cleaned_text:
                        all_text.append(cleaned_text)
        
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
    
    # Combine all text and write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(all_text))
    
    print(f"Processing complete. Output saved to {output_file}")

if __name__ == "__main__":
    # Define input and output paths
    input_folder = "../data"  # folder containing PDF files
    output_file = "../data/dataset.txt"  # output text file
    
    # Process the PDFs
    process_pdfs(input_folder, output_file)
