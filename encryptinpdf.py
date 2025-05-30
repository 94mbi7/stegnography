import os
from cryptography.fernet import Fernet

def generate_encryption_key(key_path="encryption_key.key"):
    key = Fernet.generate_key()
    with open(key_path, 'wb') as key_file:
        key_file.write(key)
    print(f"Encryption key saved to: {key_path}")
    return key

def load_encryption_key(key_path="encryption_key.key"):
    with open(key_path, 'rb') as key_file:
        return key_file.read()

def embed_video_in_pdf(pdf_path, video_path, output_pdf, encryption_key=None):

    with open(pdf_path, 'rb') as pdf_file:
        pdf_data = pdf_file.read()

    with open(video_path, 'rb') as video_file:
        video_data = video_file.read()

    if encryption_key:
        cipher = Fernet(encryption_key)
        video_data = cipher.encrypt(video_data)

    start_marker = b"\n%VIDEO_DATA_START%\n"
    end_marker = b"\n%VIDEO_DATA_END%\n"

    with open(output_pdf, 'wb') as output_file:
        output_file.write(pdf_data)
        output_file.write(start_marker)
        output_file.write(video_data)
        output_file.write(end_marker)

    print(f"Video embedded into PDF: {output_pdf}")

def extract_video_from_pdf(pdf_path, output_video, encryption_key=None):
    """
    Extracts a hidden video file from a PDF, optionally decrypting it.
    :param pdf_path: Path to the input PDF file with embedded video.
    :param output_video: Path to save the extracted video file.
    :param encryption_key: Encryption key for decryption (None if no encryption).
    """
    with open(pdf_path, 'rb') as pdf_file:
        pdf_data = pdf_file.read()

    start_marker = b"%VIDEO_DATA_START%"
    end_marker = b"%VIDEO_DATA_END%"

    start_index = pdf_data.find(start_marker) + len(start_marker)
    end_index = pdf_data.find(end_marker)

    if start_index == -1 or end_index == -1:
        print("No hidden video found in the PDF.")
        return

    video_data = pdf_data[start_index:end_index]

    if encryption_key:
        cipher = Fernet(encryption_key)
        video_data = cipher.decrypt(video_data)

    with open(output_video, 'wb') as video_file:
        video_file.write(video_data)

    print(f"Video extracted and saved to: {output_video}")

if __name__ == "__main__":

    input_pdf = "/home/gambit/Music/iukuk/y.pdf"  
    video_to_hide = "/home/gambit/Music/iukuk/part3.mp4"  
    stego_pdf = "/home/gambit/Music/iukuk/y3.pdf"  
    extracted_video = "extracted_video.mp4"  
    encryption_key_path = "encryption_key.key" 

    encryption_key = generate_encryption_key(encryption_key_path)

    embed_video_in_pdf(input_pdf, video_to_hide, stego_pdf, encryption_key)

    extract_video_from_pdf(stego_pdf, extracted_video, encryption_key)
