from cryptography.fernet import Fernet

def extract_video_from_pdf(pdf_path, output_video, encryption_key=None):

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
        try:
            video_data = cipher.decrypt(video_data)
        except Exception as e:
            print(f"Error: Failed to decrypt video data. {e}")
            return

    with open(output_video, 'wb') as video_file:
        video_file.write(video_data)

    print(f"Video extracted and saved to: {output_video}")

if __name__ == "__main__":
    stego_pdf = ""  
    extracted_video = ""  
    encryption_key_path = "encryption_key.key"  

    encryption_key = None
    try:
        with open(encryption_key_path, 'rb') as key_file:
            encryption_key = key_file.read()
    except FileNotFoundError:
        print("Encryption key not found. Continuing without decryption...")

    extract_video_from_pdf(stego_pdf, extracted_video, encryption_key)
