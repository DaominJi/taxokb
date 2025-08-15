import os
import argparse
from mistralai import Mistral
import base64

def convert_pdf_to_md(input_path, output_path, client):
    """
    Convert PDF to Markdown using Mistral OCR API
    """
    try:
        # Upload PDF file
        print(f"Uploading {input_path}...")
        with open(input_path, "rb") as f:
            uploaded_pdf = client.files.upload(
                file={
                    "file_name": os.path.basename(input_path),
                    "content": f.read(),
                },
                purpose="ocr"
            )

        # Get signed URL for OCR processing
        signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

        # Process with OCR
        print("Processing with OCR...")
        try:
            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": signed_url.url,
                },
                include_image_base64=True
            )
        except Exception as ocr_error:
            print(f"OCR processing failed: {ocr_error}")
            # Clean up uploaded file
            client.files.delete(file_id=uploaded_pdf.id)
            return False

        # Convert OCR response to Markdown
        markdown_content = ""
        
        # Add title from filename
        title = os.path.splitext(os.path.basename(input_path))[0]
        #markdown_content += f"# {title}\n\n"
        
        # Process each page
        for page_num, page in enumerate(ocr_response.pages, 1):
            #markdown_content += f"## Page {page_num}\n\n"
            
            # Check if page has markdown content directly
            if hasattr(page, 'markdown') and page.markdown:
                markdown_content += page.markdown + "\n\n"
                continue
            
            # Check if page has text content
            if hasattr(page, 'text') and page.text:
                markdown_content += page.text + "\n\n"
                continue
            
            # Process blocks if they exist
            if hasattr(page, 'blocks') and page.blocks:
                for block in page.blocks:
                    # Check if block has text content
                    if hasattr(block, 'text') and block.text:
                        # Determine heading level based on block type or formatting
                        if block.text.isupper() or len(block.text) < 50:
                            # Likely a heading
                            markdown_content += f"### {block.text}\n\n"
                        else:
                            # Regular paragraph
                            markdown_content += f"{block.text}\n\n"
                    
                    # Process lines within blocks
                    elif hasattr(block, 'lines'):
                        for line in block.lines:
                            if hasattr(line, 'text') and line.text:
                                markdown_content += f"{line.text}\n"
                        markdown_content += "\n"
                    
                    # Process words if no higher-level text is available
                    elif hasattr(block, 'words'):
                        line_text = ""
                        for word in block.words:
                            if hasattr(word, 'text') and word.text:
                                line_text += word.text + " "
                        if line_text.strip():
                            markdown_content += f"{line_text.strip()}\n"
                        markdown_content += "\n"
            
            # If no blocks, try to get text from other attributes
            elif hasattr(page, 'content') and page.content:
                markdown_content += page.content + "\n\n"
            elif hasattr(page, 'raw_text') and page.raw_text:
                markdown_content += page.raw_text + "\n\n"

        # Save to Markdown file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        print(f"Successfully converted {input_path} to {output_path}")
        
        # Clean up uploaded file
        client.files.delete(file_id=uploaded_pdf.id)
        print(f"Cleaned up uploaded file: {uploaded_pdf.id}")
        
        return True

    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return False

def main():
    """
    Main function to handle command-line arguments and initiate the conversion.
    """
    parser = argparse.ArgumentParser(description="Convert a PDF file to a Markdown file using the Mistral OCR API.")
    parser.add_argument("--input_path", type=str, default='data/test', help="The path to the input PDF file.")
    parser.add_argument("--output_path", type=str, nargs='?', default='data/test', help="The path to save the output Markdown file (optional).")

    args = parser.parse_args()

    # Get API key from environment variable
    api_key = os.environ.get("MISTRAL_API_KEY")

    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable not set.")
        print("Please get your API key from https://console.mistral.ai/ and set it as an environment variable.")
        return

    # Initialize Mistral client
    client = Mistral(api_key=api_key)

    pdf_path = args.input_path + '/pdf_files'
    output_path = args.output_path + '/md_files'

    if not output_path:
        # If no output path is provided, create one based on the input file's name
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_path = f"{base_name}.md"

    if not os.path.exists(pdf_path):
        print(f"Error: The file '{pdf_path}' does not exist.")
        return
    
    # Check if pdf_path is a directory
    if os.path.isdir(pdf_path):
        # Process all PDF files in directory
        for file in os.listdir(pdf_path):
            if file.endswith('.pdf'):
                full_pdf_path = os.path.join(pdf_path, file)
                full_output_path = os.path.join(output_path, file.replace('.pdf', '.md'))
                
                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
                
                convert_pdf_to_md(full_pdf_path, full_output_path, client)
                print("-"*100)
                print(f"Converted {full_pdf_path} to {full_output_path}")
                print("-"*100)
    else:
        # Process single file
        convert_pdf_to_md(pdf_path, output_path, client)

if __name__ == "__main__":
    main()
