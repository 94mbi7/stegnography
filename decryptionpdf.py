from cryptography.fernet import Fernet

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
        try:
            video_data = cipher.decrypt(video_data)
        except Exception as e:
            print(f"Error: Failed to decrypt video data. {e}")
            return

    # Write the extracted video to a file
    with open(output_video, 'wb') as video_file:
        video_file.write(video_data)

    print(f"Video extracted and saved to: {output_video}")

if __name__ == "__main__":
    # Paths
    stego_pdf = ""  # Input PDF with hidden video
    extracted_video = ""  # Path to save the extracted video
    encryption_key_path = "encryption_key.key"  # Path to the encryption key file

    # Load the encryption key (used during embedding)
    encryption_key = None
    try:
        with open(encryption_key_path, 'rb') as key_file:
            encryption_key = key_file.read()
    except FileNotFoundError:
        print("Encryption key not found. Continuing without decryption...")

    # Extract the video
    extract_video_from_pdf(stego_pdf, extracted_video, encryption_key)
