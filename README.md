# Automate PDF Processing  

## **Introduction**  

This script automates the processing of PDF documents by performing the following tasks:  

- **OCR (Optical Character Recognition):** Extracts text from PDF files using Tesseract.  
- **Language Detection:** Identifies the language of the extracted text.  
- **Translation (Optional):** Translates the text to English if it is in another language.  
- **Question Answering:** Utilizes a pre-trained BERT model to answer specific questions about the document content.  
- **Summarization:** Summarizes the document's key points using a BART-based summarizer.  
- **Information Extraction:** Extracts specific information such as company name and identifier.  

---

## **Usage**  

### **1. Process a Single PDF**  
Run the script for a specific input PDF file and save the output to a JSON file:  
```bash
python Automate.py -i <input_pdf_file> -o <output_json_file>
```  
- Replace `<input_pdf_file>` with the name of the PDF file to process.  
- Replace `<output_json_file>` with the name of the JSON file where the extracted information will be saved.  
- Ensure the input file is in the same directory as the script, or provide the full path to the file.  

### **2. Process All PDFs in a Folder**  
If no `-i` and `-o` options are provided, the script will process all PDF files in the default `BE_GAZETTE_PDFS` folder.  
```bash
python Automate.py
```  
The extracted information for each document will be saved in individual JSON files within an output folder.  

---

## **Requirements**  

The script requires the following Python libraries:  
- `torch`  
- `pytesseract`  
- `opencv-python`  
- `numpy`  
- `Pillow`  
- `langdetect`  
- `pdf2image`  
- `transformers`  

### Installation  
Install the required libraries using pip:  
```bash
pip install torch pytesseract opencv-python numpy pillow langdetect pdf2image transformers
```  

---

## **Instructions**  

1. **Install Dependencies:**  
   Ensure all required libraries are installed.  

2. **Set Up Tesseract:**  
   Install Tesseract OCR and ensure it's added to your system's PATH.  

3. **Prepare PDF Files:**  
   - For single-file processing, place the PDF file in the same directory as the script or provide its full path.  
   - For batch processing, place all the PDF files in the `BE_GAZETTE_PDFS` folder.  

4. **Run the Script:**  
   Use one of the methods outlined in the **Usage** section to process the PDFs.  

5. **Review Output:**  
   - For single-file processing, the output will be saved as the specified JSON file.  
   - For batch processing, output JSON files will be created in an automatically generated folder.  

---

## **Notes**  

- **Language Support:**  
   This script is designed for multilingual documents. Non-English documents will be automatically translated to English before further processing.  

- **Document Quality:**  
   The accuracy of OCR, translation, and information extraction depends on the quality of the input PDF and the OCR engine.  

---

## **Further Enhancements**  

Future improvements may include:  
- Enhanced error handling for OCR, translation, and NLP tasks.  
- Support for custom target languages in translation.  
- Advanced NLP techniques tailored to specific document types.  

---  

This project simplifies the extraction of valuable information from complex PDF documents. Enjoy using it! 🎉  
