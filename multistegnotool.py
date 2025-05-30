import numpy as np
import cv2
from PIL import Image
import wave
import struct
import os
import base64
import hashlib
from typing import Tuple, Optional, List
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SteganographyError(Exception):
    """Custom exception for steganography operations"""
    pass

class ImageSteganography:
    """Handles image steganography using LSB and DCT methods"""
    
    def __init__(self):
        self.delimiter = "===END_OF_MESSAGE==="
        
    def text_to_binary(self, text: str) -> str:
        """Convert text to binary representation"""
        return ''.join(format(ord(char), '08b') for char in text)
    
    def binary_to_text(self, binary: str) -> str:
        """Convert binary to text"""
        text = ''
        for i in range(0, len(binary), 8):
            byte = binary[i:i+8]
            if len(byte) == 8:
                try:
                    text += chr(int(byte, 2))
                except ValueError:
                    continue
        return text
    
    def embed_lsb(self, image_path: str, secret_message: str, output_path: str, password: str = None) -> dict:
        """Embed secret message using LSB steganography"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise SteganographyError(f"Could not load image: {image_path}")
            
            if password:
                secret_message = self._encrypt_message(secret_message, password)
            
            message_with_delimiter = secret_message + self.delimiter
            binary_message = self.text_to_binary(message_with_delimiter)
            
            total_pixels = img.shape[0] * img.shape[1] * img.shape[2]
            if len(binary_message) > total_pixels:
                raise SteganographyError("Message too large for image capacity")
            
            data_index = 0
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    for k in range(img.shape[2]):
                        if data_index < len(binary_message):
                            img[i, j, k] = (img[i, j, k] & 0xFE) | int(binary_message[data_index])
                            data_index += 1
                        else:
                            break
                    if data_index >= len(binary_message):
                        break
                if data_index >= len(binary_message):
                    break
            
            cv2.imwrite(output_path, img)
            
            original_img = cv2.imread(image_path)
            psnr = self._calculate_psnr(original_img, img)
            mse = self._calculate_mse(original_img, img)
            
            return {
                'success': True,
                'output_path': output_path,
                'message_length': len(secret_message),
                'capacity_used': len(binary_message) / total_pixels * 100,
                'psnr': psnr,
                'mse': mse
            }
            
        except Exception as e:
            logger.error(f"LSB embedding failed: {str(e)}")
            raise SteganographyError(f"LSB embedding failed: {str(e)}")
    
    def extract_lsb(self, stego_image_path: str, password: str = None) -> str:
        """Extract secret message using LSB steganography"""
        try:
            img = cv2.imread(stego_image_path)
            if img is None:
                raise SteganographyError(f"Could not load image: {stego_image_path}")
            
            binary_message = ""
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    for k in range(img.shape[2]):
                        binary_message += str(img[i, j, k] & 1)
            
            message = self.binary_to_text(binary_message)
            delimiter_index = message.find(self.delimiter)
            
            if delimiter_index == -1:
                raise SteganographyError("No hidden message found or message corrupted")
            
            extracted_message = message[:delimiter_index]
            
            if password:
                extracted_message = self._decrypt_message(extracted_message, password)
            
            return extracted_message
            
        except Exception as e:
            logger.error(f"LSB extraction failed: {str(e)}")
            raise SteganographyError(f"LSB extraction failed: {str(e)}")
    
    def embed_dct(self, image_path: str, secret_message: str, output_path: str, 
                  alpha: float = 10.0, password: str = None) -> dict:
        """Embed secret message using DCT method with password support"""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise SteganographyError(f"Could not load image: {image_path}")
            
            # Apply password encryption if provided
            if password:
                secret_message = self._encrypt_message(secret_message, password)
            
            message_with_delimiter = secret_message + self.delimiter
            binary_message = self.text_to_binary(message_with_delimiter)
            
            img_float = np.float32(img)
            blocks = self._divide_into_blocks(img_float, 8)
            
            if len(binary_message) > len(blocks):
                raise SteganographyError("Message too large for DCT embedding")
            
            modified_blocks = []
            for i, block in enumerate(blocks):
                if i < len(binary_message):
                    dct_block = cv2.dct(block)
                    
                    # Use a more robust embedding strategy
                    bit = int(binary_message[i])
                    if bit == 1:
                        dct_block[4, 4] = abs(dct_block[4, 4]) + alpha
                    else:
                        dct_block[4, 4] = -(abs(dct_block[4, 4]) + alpha)
                    
                    modified_block = cv2.idct(dct_block)
                    modified_blocks.append(modified_block)
                else:
                    modified_blocks.append(block)
            
            reconstructed_img = self._reconstruct_from_blocks(modified_blocks, img.shape, 8)
            reconstructed_img = np.clip(reconstructed_img, 0, 255).astype(np.uint8)
            
            cv2.imwrite(output_path, reconstructed_img)
            
            psnr = self._calculate_psnr(img, reconstructed_img)
            mse = self._calculate_mse(img, reconstructed_img)
            
            return {
                'success': True,
                'output_path': output_path,
                'message_length': len(secret_message),
                'psnr': psnr,
                'mse': mse,
                'alpha': alpha
            }
            
        except Exception as e:
            logger.error(f"DCT embedding failed: {str(e)}")
            raise SteganographyError(f"DCT embedding failed: {str(e)}")
    
    def extract_dct(self, stego_image_path: str, alpha: float = 10.0, password: str = None) -> str:
        """Extract secret message using DCT method with password support"""
        try:
            img = cv2.imread(stego_image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise SteganographyError(f"Could not load image: {stego_image_path}")
            
            img_float = np.float32(img)
            blocks = self._divide_into_blocks(img_float, 8)
            
            binary_message = ""
            for block in blocks:
                dct_block = cv2.dct(block)
                # More robust extraction logic
                if dct_block[4, 4] > 0:
                    binary_message += "1"
                else:
                    binary_message += "0"
            
            message = self.binary_to_text(binary_message)
            delimiter_index = message.find(self.delimiter)
            
            if delimiter_index == -1:
                raise SteganographyError("No hidden message found or message corrupted")
            
            extracted_message = message[:delimiter_index]
            
            # Apply password decryption if provided
            if password:
                extracted_message = self._decrypt_message(extracted_message, password)
            
            return extracted_message
            
        except Exception as e:
            logger.error(f"DCT extraction failed: {str(e)}")
            raise SteganographyError(f"DCT extraction failed: {str(e)}")
    
    def _divide_into_blocks(self, img: np.ndarray, block_size: int) -> List[np.ndarray]:
        """Divide image into blocks"""
        blocks = []
        h, w = img.shape
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = img[i:i+block_size, j:j+block_size]
                if block.shape == (block_size, block_size):
                    blocks.append(block)
        return blocks
    
    def _reconstruct_from_blocks(self, blocks: List[np.ndarray], img_shape: Tuple, block_size: int) -> np.ndarray:
        """Reconstruct image from blocks"""
        h, w = img_shape
        reconstructed = np.zeros((h, w), dtype=np.float32)
        block_idx = 0
        
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                if block_idx < len(blocks):
                    reconstructed[i:i+block_size, j:j+block_size] = blocks[block_idx]
                    block_idx += 1
        
        return reconstructed
    
    def _calculate_psnr(self, original: np.ndarray, modified: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = np.mean((original - modified) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    def _calculate_mse(self, original: np.ndarray, modified: np.ndarray) -> float:
        """Calculate Mean Squared Error"""
        return np.mean((original - modified) ** 2)
    
    def _encrypt_message(self, message: str, password: str) -> str:
        """Enhanced XOR encryption with base64 encoding"""
        try:
            # Create a key from password using hash
            key = hashlib.sha256(password.encode()).hexdigest()
            encrypted = ""
            for i, char in enumerate(message):
                key_char = key[i % len(key)]
                encrypted += chr(ord(char) ^ ord(key_char))
            return base64.b64encode(encrypted.encode('latin-1')).decode()
        except Exception as e:
            raise SteganographyError(f"Encryption failed: {str(e)}")
    
    def _decrypt_message(self, encrypted_message: str, password: str) -> str:
        """Enhanced XOR decryption with base64 decoding"""
        try:
            # Create the same key from password
            key = hashlib.sha256(password.encode()).hexdigest()
            encrypted = base64.b64decode(encrypted_message.encode()).decode('latin-1')
            decrypted = ""
            for i, char in enumerate(encrypted):
                key_char = key[i % len(key)]
                decrypted += chr(ord(char) ^ ord(key_char))
            return decrypted
        except Exception as e:
            raise SteganographyError("Failed to decrypt message - incorrect password or corrupted data")

class SteganographyToolkit:
    """Main toolkit class that integrates all steganography methods"""
    
    def __init__(self):
        self.image_stego = ImageSteganography()
        
    def embed(self, cover_path: str, secret_message: str, output_path: str, 
              method: str = 'lsb', password: str = None, **kwargs) -> dict:
        """Universal embed function"""
        try:
            file_ext = os.path.splitext(cover_path)[1].lower()
            
            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                if method == 'lsb':
                    return self.image_stego.embed_lsb(cover_path, secret_message, output_path, password)
                elif method == 'dct':
                    alpha = kwargs.get('alpha', 10.0)
                    return self.image_stego.embed_dct(cover_path, secret_message, output_path, alpha, password)
                else:
                    raise SteganographyError(f"Unsupported image method: {method}")
            else:
                raise SteganographyError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            logger.error(f"Embedding failed: {str(e)}")
            raise
    
    def extract(self, stego_path: str, method: str = 'lsb', password: str = None, **kwargs) -> str:
        """Universal extract function"""
        try:
            file_ext = os.path.splitext(stego_path)[1].lower()
            
            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                if method == 'lsb':
                    return self.image_stego.extract_lsb(stego_path, password)
                elif method == 'dct':
                    alpha = kwargs.get('alpha', 10.0)
                    return self.image_stego.extract_dct(stego_path, alpha, password)
                else:
                    raise SteganographyError(f"Unsupported image method: {method}")
            else:
                raise SteganographyError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            raise
    
    def analyze_capacity(self, cover_path: str) -> dict:
        """Analyze the hiding capacity of a cover file"""
        try:
            file_ext = os.path.splitext(cover_path)[1].lower()
            
            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                img = cv2.imread(cover_path)
                if img is None:
                    raise SteganographyError(f"Could not load image: {cover_path}")
                
                total_pixels = img.shape[0] * img.shape[1] * img.shape[2]
                lsb_capacity_bits = total_pixels
                lsb_capacity_chars = lsb_capacity_bits // 8
                
                # DCT capacity (8x8 blocks)
                blocks_h = img.shape[0] // 8
                blocks_w = img.shape[1] // 8
                dct_blocks = blocks_h * blocks_w
                dct_capacity_chars = dct_blocks // 8
                
                return {
                    'file_type': 'image',
                    'dimensions': f"{img.shape[1]}x{img.shape[0]}",
                    'channels': img.shape[2],
                    'total_pixels': total_pixels,
                    'lsb_capacity_chars': lsb_capacity_chars,
                    'lsb_capacity_kb': lsb_capacity_chars / 1024,
                    'dct_capacity_chars': dct_capacity_chars,
                    'dct_capacity_kb': dct_capacity_chars / 1024
                }
            else:
                raise SteganographyError(f"Unsupported file format for analysis: {file_ext}")
                
        except Exception as e:
            logger.error(f"Capacity analysis failed: {str(e)}")
            raise
    
    def detect_steganography(self, file_path: str) -> dict:
        """Basic steganalysis to detect potential steganography"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                return self._detect_image_stego(file_path)
            else:
                return {'detected': False, 'reason': 'Unsupported format for detection'}
                
        except Exception as e:
            logger.error(f"Steganalysis failed: {str(e)}")
            return {'detected': False, 'error': str(e)}

    def _detect_image_stego(self, image_path: str) -> dict:
        """Detect steganography in images using statistical analysis"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise SteganographyError(f"Could not load image: {image_path}")
            
            chi_square_score = self._chi_square_test(img)
            lsb_histogram_anomaly = self._lsb_histogram_analysis(img)
            
            suspicion_score = (chi_square_score + lsb_histogram_anomaly) / 2
            
            return {
                'detected': bool(suspicion_score > 0.7),
                'suspicion_score': float(suspicion_score),
                'chi_square_score': float(chi_square_score),
                'lsb_histogram_anomaly': float(lsb_histogram_anomaly),
                'analysis': 'High suspicion' if suspicion_score > 0.8 else 'Medium suspicion' if suspicion_score > 0.5 else 'Low suspicion'
            }
            
        except Exception as e:
            return {'detected': False, 'error': str(e)}

    def _chi_square_test(self, img: np.ndarray) -> float:
        """Chi-square test for detecting LSB steganography"""
        try:
            flat_img = img.flatten()
            
            pairs = []
            for i in range(0, 256, 2):
                count_p = np.sum(flat_img == i)
                count_p1 = np.sum(flat_img == i + 1) if i + 1 < 256 else 0
                pairs.append((count_p, count_p1))
            
            chi_square = 0
            for count_p, count_p1 in pairs:
                expected = (count_p + count_p1) / 2
                if expected > 0:
                    chi_square += ((count_p - expected) ** 2) / expected
                    chi_square += ((count_p1 - expected) ** 2) / expected
            
            normalized_score = min(chi_square / 1000, 1.0)
            return float(normalized_score)
            
        except Exception:
            return 0.0

    def _lsb_histogram_analysis(self, img: np.ndarray) -> float:
        """Analyze LSB histogram for anomalies"""
        try:
            flat_img = img.flatten()
            
            lsb_0_count = np.sum((flat_img & 1) == 0)
            lsb_1_count = np.sum((flat_img & 1) == 1)
            
            total_pixels = len(flat_img)
            lsb_0_ratio = lsb_0_count / total_pixels
            
            deviation = abs(lsb_0_ratio - 0.5)
            suspicion_score = min(deviation * 4, 1.0)
            return float(suspicion_score)
            
        except Exception:
            return 0.0

class SteganographyGUI:
    """Simple GUI interface for the toolkit"""
    
    def __init__(self):
        self.toolkit = SteganographyToolkit()
        
    def create_gui(self):
        """Create a simple command-line interface"""
        while True:
            print("\n" + "="*50)
            print("STEGANOGRAPHY TOOLKIT")
            print("="*50)
            print("1. Embed message")
            print("2. Extract message")
            print("3. Analyze capacity")
            print("4. Detect steganography")
            print("5. Exit")
            print("-"*50)
            
            choice = input("Select option (1-5): ").strip()
            
            if choice == '1':
                self._embed_interface()
            elif choice == '2':
                self._extract_interface()
            elif choice == '3':
                self._analyze_interface()
            elif choice == '4':
                self._detect_interface()
            elif choice == '5':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")
    
    def _embed_interface(self):
        """Interface for embedding messages"""
        try:
            print("\n--- EMBED MESSAGE ---")
            cover_path = input("Enter cover file path: ").strip()
            if not os.path.exists(cover_path):
                print("File not found!")
                return
            
            message = input("Enter secret message: ").strip()
            output_path = input("Enter output file path: ").strip()
            method = input("Enter method (lsb/dct): ").strip().lower()
            if not method:
                method = 'lsb'
            
            password = input("Enter password (optional): ").strip()
            if not password:
                password = None
            
            print("Embedding...")
            result = self.toolkit.embed(cover_path, message, output_path, method, password)
            print(f"Success! Result: {result}")
            
        except Exception as e:
            print(f"Embedding failed: {e}")
    
    def _extract_interface(self):
        """Interface for extracting messages"""
        try:
            print("\n--- EXTRACT MESSAGE ---")
            stego_path = input("Enter stego file path: ").strip()
            if not os.path.exists(stego_path):
                print("File not found!")
                return
            
            method = input("Enter method (lsb/dct): ").strip().lower()
            if not method:
                method = 'lsb'
            
            password = input("Enter password (if used): ").strip()
            if not password:
                password = None
            
            print("Extracting...")
            message = self.toolkit.extract(stego_path, method, password)
            print(f"Extracted message: {message}")
            
        except Exception as e:
            print(f"Extraction failed: {e}")
    
    def _analyze_interface(self):
        """Interface for capacity analysis"""
        try:
            print("\n--- ANALYZE CAPACITY ---")
            file_path = input("Enter file path: ").strip()
            if not os.path.exists(file_path):
                print("File not found!")
                return
            
            print("Analyzing...")
            capacity = self.toolkit.analyze_capacity(file_path)
            print(f"Capacity analysis: {json.dumps(capacity, indent=2)}")
            
        except Exception as e:
            print(f"Analysis failed: {e}")
    
    def _detect_interface(self):
        """Interface for steganalysis"""
        try:
            print("\n--- DETECT STEGANOGRAPHY ---")
            file_path = input("Enter file path: ").strip()
            if not os.path.exists(file_path):
                print("File not found!")
                return
            
            print("Analyzing...")
            detection = self.toolkit.detect_steganography(file_path)
            print(f"Detection result: {json.dumps(detection, indent=2)}")
            
        except Exception as e:
            print(f"Detection failed: {e}")

if __name__ == "__main__":
    gui = SteganographyGUI()
    gui.create_gui()
