from cryptography.fernet import Fernet
# NOTE: for real production do not generate key at runtime; use KMS
KEY = Fernet.generate_key()
cipher = Fernet(KEY)

def encrypt(x: bytes) -> bytes:
    return cipher.encrypt(x)

def decrypt(x: bytes) -> bytes:
    return cipher.decrypt(x)
