import os
from cryptography.fernet import Fernet

def generate_encryption_key(key_path="encryption_key.key"):
    """
    Generates a new encryption key and saves it to a file.
    :param key_path: Path to save the encryption key.
    :return: The generated encryption key.
    """
    key = Fernet.generate_key()
    with open(key_path, 'wb') as key_file:
        key_file.write(key)
    print(f"Encryption key saved to: {key_path}")
    return key

def load_encryption_key(key_path="encryption_key.key"):
    """
    Loads an encryption key from a file.
    :param key_path: Path to the encryption key file.
    :return: The encryption key.
    """
    with open(key_path, 'rb') as key_file:
        return key_file.read()

def embed_video_in_pdf(pdf_path, video_path, output_pdf, encryption_key=None):
    """
    Embeds a video file into a PDF by appending it as hidden data, optionally encrypted.
    :param pdf_path: Path to the input PDF file.
    :param video_path: Path to the video file to embed.
    :param output_pdf: Path to save the output PDF with embedded video.
    :param encryption_key: Encryption key for optional encryption (None if no encryption).
    """
    # Read the PDF file
    with open(pdf_path, 'rb') as pdf_file:
        pdf_data = pdf_file.read()

    # Read the video file
    with open(video_path, 'rb') as video_file:
        video_data = video_file.read()

    # Encrypt the video data if an encryption key is provided
    if encryption_key:
        cipher = Fernet(encryption_key)
        video_data = cipher.encrypt(video_data)

    # Define markers to identify the hidden video data
    start_marker = b"\n%VIDEO_DATA_START%\n"
    end_marker = b"\n%VIDEO_DATA_END%\n"

    # Append the video data to the PDF
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
    # Read the PDF file
    with open(pdf_path, 'rb') as pdf_file:
        pdf_data = pdf_file.read()

    # Define markers to identify the hidden video data
    start_marker = b"%VIDEO_DATA_START%"
    end_marker = b"%VIDEO_DATA_END%"

    # Locate the start and end markers
    start_index = pdf_data.find(start_marker) + len(start_marker)
    end_index = pdf_data.find(end_marker)

    # Check if the markers are found
    if start_index == -1 or end_index == -1:
        print("No hidden video found in the PDF.")
        return

    # Extract the video data
    video_data = pdf_data[start_index:end_index]

    # Decrypt the video data if an encryption key is provided
    if encryption_key:
        cipher = Fernet(encryption_key)
        video_data = cipher.decrypt(video_data)

    # Write the extracted video to a file
    with open(output_video, 'wb') as video_file:
        video_file.write(video_data)

    print(f"Video extracted and saved to: {output_video}")

if __name__ == "__main__":
    # Example Usage

    # Paths
    input_pdf = "/home/gambit/Music/iukuk/y.pdf"  # Path to the input PDF
    video_to_hide = "/home/gambit/Music/iukuk/part3.mp4"  # Path to the video file to embed
    stego_pdf = "/home/gambit/Music/iukuk/y3.pdf"  # Output PDF with hidden video
    extracted_video = "extracted_video.mp4"  # Output path for the extracted video
    encryption_key_path = "encryption_key.key"  # Path to the encryption key file

    # Generate an encryption key (run this once and keep the key file safe)
    encryption_key = generate_encryption_key(encryption_key_path)

    # Alternatively, load an existing encryption key
    # encryption_key = load_encryption_key(encryption_key_path)

    # Embed the video into the PDF
    embed_video_in_pdf(input_pdf, video_to_hide, stego_pdf, encryption_key)

    # Extract the video from the PDF
    extract_video_from_pdf(stego_pdf, extracted_video, encryption_key)
