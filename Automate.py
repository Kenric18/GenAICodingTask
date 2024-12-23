import os
import json
import torch
import pytesseract
import cv2
import numpy as np
from PIL import Image
from langdetect import detect
from pdf2image import convert_from_path
from transformers import MarianMTModel, MarianTokenizer, AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import argparse

# Initialize device for GPU/CPU use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the models for summarization, translation, and question answering
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt", device=device)
    qa_model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    qa_tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer, device=device)
    return summarizer, qa_pipeline


summarizer, qa_pipeline = load_models()


# Step 1: Preprocess the image to improve OCR accuracy
def preprocess_image(image):
    grey_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, thresholded_image = cv2.threshold(grey_image, 150, 255, cv2.THRESH_BINARY_INV)
    denoised_image = cv2.GaussianBlur(thresholded_image, (5, 5), 0)
    dilated_image = cv2.dilate(denoised_image, (2, 2), iterations=1)
    eroded_image = cv2.erode(dilated_image, (2, 2), iterations=1)
    return Image.fromarray(eroded_image)


# Step 2: Extract text from PDF using OCR
def extract_text_from_pdf(pdf_path, save_path=""):
    images = convert_from_path(pdf_path)
    text = ""
    for i, img in enumerate(images):
        width, height = img.size
        crop_box = (0.15 * width, 0.18 * height, width, height) if i == 0 else (0.15 * width, 0, width, height)
        cropped_image = img.crop(crop_box)
        if save_path:
            cropped_image.save(f"{save_path}/Page_{i}.jpg")
        preprocessed_image = preprocess_image(cropped_image)
        text += pytesseract.image_to_string(preprocessed_image, lang='fra+nld+deu') + "\n"
    return text


# Step 3: Detect the language of the extracted text
def detect_language(text):
    return detect(text)


# Step 4: Translate text if not in English
def translate_text(text, source_lang, target_lang="en", save_path=""):
    if source_lang == "en":  # Skip translation if already English
        return text

    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)

    translated_text = ""
    for chunk in text.split('\n'):
        if chunk.strip():
            inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True).to(device)
            translated_tokens = model.generate(**inputs)
            translated_chunk = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            translated_text += translated_chunk + "\n"

    if save_path:
        with open(f"{save_path}/translated.txt", "w", encoding="utf-8") as file:
            file.write(translated_text)

    return translated_text


# Step 5: Use BERT for Question Answering
def answer_question_with_context(context, prompt):
    result = qa_pipeline(question=prompt, context=context)
    return result["answer"]


# Step 6: Summarize the document
def summarize_document(text, max_length=1024):
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    sentences = text.split("\n")
    chunk = ""
    chunks = []

    # Split text into chunks based on max token length (approximately 1024 tokens)
    for sentence in sentences:
        chunk += sentence + "\n"
        # Check if the chunk exceeds max_length in terms of tokens
        tokenized_chunk = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        if tokenized_chunk['input_ids'].size(1) > max_length:
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk.strip())
            chunk = ""

    # Add the last chunk if it's not empty
    if chunk.strip():
        chunks.append(chunk.strip())

    summaries = []
    for c in chunks:
        if not c.strip():
            continue
        try:
            # Ensure the chunk is not too short (for better summarization)
            if len(c.split()) > 10:  # Only summarize if the chunk has more than 10 words
                summary = summarizer(c, max_length=200, min_length=50, do_sample=False)
                if summary:
                    summaries.append(summary[0]['summary_text'])
            else:
                # If the chunk is too short, just append it as-is
                summaries.append(c)
        except Exception as e:
            print(f"Error summarizing chunk: {str(e)}")

    # Combine all summaries into one
    full_summary = " ".join(summaries)
    return full_summary


# Step 7: Use LLM to extract information
def extract_information(context):
    print(f'Applying NLP Techniques')
    questions = {
        "Company Name": "What is the company name mentioned in the document?",
        "Company Identifier": "What is the company identifier (e.g., 8 or 10 digit number)?",
        "Document Purpose": "What is the purpose of the document?",
    }

    extracted_info = {}
    for key, question in questions.items():
        try:
            extracted_info[key] = answer_question_with_context(context, question)
        except Exception as e:
            extracted_info[key] = f"Error: {str(e)}"

    extracted_info["Summary"] = summarize_document(context)
    return extracted_info


# Main function to process the input PDF file and output the extracted information
def process_pdf(input_path, output_path):
    # Extract text from the PDF
    print(f'Extracting text from pdf')
    text = extract_text_from_pdf(input_path)

    # Detect language of the extracted text
    detected_lang = detect_language(text)
    print(f"Detected Language: {detected_lang}")

    # Translate text if necessary
    if detected_lang != "en":
        print(f'Translating to English')
        text = translate_text(text, detected_lang)

    # Extract information
    extracted_info = extract_information(text)
    print(extracted_info)

    # Save the extracted information to a JSON file
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(extracted_info, json_file, indent=4)

# Process all PDF files in a folder
def process_folder(folder_path, output_folder):
    items = os.listdir(folder_path)
    files = [os.path.join(folder_path, item) for item in items if item.endswith(".pdf")]

    for i, file in enumerate(files):
        print(f"Processing document: {os.path.basename(file)}")
        output_path = os.path.join(output_folder, f"document_{i+1}_info.json")
        process_pdf(file, output_path)

# Command-line interface for the script
def main(input_pdf=None, output_json=None):
    if input_pdf and output_json:
        # Command line argument processing
        process_pdf(input_pdf, output_json)
    else:
        # Folder processing
        folder_path = "BE_GAZETTE_PDFS"
        items = os.listdir(folder_path)
        pdf_files = [os.path.join(folder_path, item) for item in items if item.endswith('.pdf')]

        for i, file in enumerate(pdf_files):
            output_folder = f"output_{i + 1}"
            os.makedirs(output_folder, exist_ok=True)

            output_json = os.path.join(output_folder, "extracted_info.json")
            print(f"Processing {file}")
            process_pdf(file, output_json)


# Run the script if executed from the command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and extract information from a PDF")
    parser.add_argument("-i", "--input", help="Input PDF file path")
    parser.add_argument("-o", "--output", help="Output JSON file path")
    args = parser.parse_args()

    main(args.input, args.output)
