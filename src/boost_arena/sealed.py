"""Sealing a model file so that only the scoring service can open it.

An entrant seals their model with the project's public key and can then put the sealed
file anywhere public. Only the holder of the matching private key, the scoring service,
can open it. The project never publishes the model itself.

The scheme is a standard sealed box: a fresh X25519 key pair for every file, a shared
secret with the project's key, a key derived from it with HKDF-SHA256, and the model
encrypted with ChaCha20-Poly1305. The file's header is authenticated along with the data.

A sealed file is bound to one submission: the slug and the GitHub login it was sealed for
are authenticated with it, so a file copied into another entrant's submission does not open.
"""

import base64
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

MAGIC = b"BOOSTARENA-SEALED-2\n"
KEY_BYTES = 32
NONCE_BYTES = 12
INFO = b"boost-arena sealed model v2"


class SealedFileError(Exception):
    """The file is not a sealed model, or was not sealed for this key."""


def generate_key_pair():
    """A new (private, public) key pair, both as text."""
    private = X25519PrivateKey.generate()
    return encode_private_key(private), encode_public_key(private.public_key())


def encode_public_key(key: X25519PublicKey) -> str:
    raw = key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode("ascii")


def encode_private_key(key: X25519PrivateKey) -> str:
    raw = key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
    return base64.b64encode(raw).decode("ascii")


def decode_public_key(text: str) -> X25519PublicKey:
    try:
        raw = base64.b64decode(text.strip(), validate=True)
        if len(raw) != KEY_BYTES:
            raise ValueError
        return X25519PublicKey.from_public_bytes(raw)
    except Exception:
        raise SealedFileError("That is not a valid public key") from None


def decode_private_key(text: str) -> X25519PrivateKey:
    try:
        raw = base64.b64decode(text.strip(), validate=True)
        if len(raw) != KEY_BYTES:
            raise ValueError
        return X25519PrivateKey.from_private_bytes(raw)
    except Exception:
        raise SealedFileError("That is not a valid private key") from None


def _derive_key(shared_secret: bytes, sender_public: bytes, receiver_public: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=KEY_BYTES, salt=None, info=INFO + sender_public + receiver_public).derive(
        shared_secret
    )


def _binding(slug: str, github: str) -> bytes:
    """What a sealed file is tied to, in a form that cannot be confused by moving a boundary."""
    if not slug or not github or "\0" in slug or "\0" in github:
        raise SealedFileError("A sealed file must be bound to a slug and a GitHub login")
    return b"\0" + slug.encode("utf-8") + b"\0" + github.encode("utf-8") + b"\0"


def seal(data: bytes, public_key_text: str, slug: str, github: str) -> bytes:
    """Seals `data` for the holder of the private key matching `public_key_text`, bound to one submission."""
    receiver = decode_public_key(public_key_text)
    receiver_raw = receiver.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    sender = X25519PrivateKey.generate()
    sender_raw = sender.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    key = _derive_key(sender.exchange(receiver), sender_raw, receiver_raw)

    nonce = os.urandom(NONCE_BYTES)
    header = MAGIC + sender_raw + nonce
    return header + ChaCha20Poly1305(key).encrypt(nonce, data, header + _binding(slug, github))


def is_sealed(data: bytes) -> bool:
    return data.startswith(MAGIC) and len(data) > len(MAGIC) + KEY_BYTES + NONCE_BYTES + 16


def open_sealed(data: bytes, private_key_text: str, slug: str, github: str) -> bytes:
    """Opens a sealed file. Raises SealedFileError if it was not sealed for this key and this
    submission, or was tampered with."""
    if not is_sealed(data):
        raise SealedFileError("This is not a sealed model file")

    receiver = decode_private_key(private_key_text)
    receiver_raw = receiver.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    offset = len(MAGIC)
    sender_raw = data[offset : offset + KEY_BYTES]
    nonce = data[offset + KEY_BYTES : offset + KEY_BYTES + NONCE_BYTES]
    header = data[: offset + KEY_BYTES + NONCE_BYTES]
    body = data[len(header) :]

    try:
        sender = X25519PublicKey.from_public_bytes(sender_raw)
        key = _derive_key(receiver.exchange(sender), sender_raw, receiver_raw)
        return ChaCha20Poly1305(key).decrypt(nonce, body, header + _binding(slug, github))
    except InvalidTag:
        raise SealedFileError(
            "The file was not sealed for this key and this submission, or has been changed since it was sealed"
        ) from None
    except SealedFileError:
        raise
    except Exception:
        raise SealedFileError("The sealed file could not be opened") from None
