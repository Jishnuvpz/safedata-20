import hashlib
import secrets
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class SecurityManager:
    """Security utilities for SafeData 2.0"""
    
    def __init__(self):
        self.secret_key = os.getenv("SECRET_KEY", "default-secret-key").encode()
        
    def generate_encryption_key(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """Generate encryption key from password"""
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt_data(self, data: str, key: bytes) -> str:
        """Encrypt sensitive data"""
        f = Fernet(key)
        encrypted_data = f.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt_data(self, encrypted_data: str, key: bytes) -> str:
        """Decrypt sensitive data"""
        f = Fernet(key)
        decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted_data = f.decrypt(decoded_data)
        return decrypted_data.decode()
    
    def hash_data(self, data: str) -> str:
        """Create secure hash of data"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def generate_session_token(self) -> str:
        """Generate secure session token"""
        return secrets.token_urlsafe(32)
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize uploaded filename"""
        # Remove path components
        filename = os.path.basename(filename)
        # Remove dangerous characters
        filename = "".join(c for c in filename if c.isalnum() or c in "._-")
        return filename
    
    def validate_file_type(self, filename: str, allowed_extensions: list) -> bool:
        """Validate file type against allowed extensions"""
        return any(filename.lower().endswith(ext) for ext in allowed_extensions)

security_manager = SecurityManager()
