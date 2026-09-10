"""End-to-end encryption: X25519 key exchange + ChaCha20-Poly1305 AEAD.

Uses `cryptography` (pyca, Apache-2.0/BSD-3-Clause, the standard audited
Python crypto library) -- hand-rolling any crypto primitive here would be
reckless regardless of how simple it looks.

X25519 is a key-AGREEMENT scheme, not "encrypt with public key, decrypt
with private key" like RSA: both sides run ECDH using (their own private
key + the peer's public key) and derive the SAME shared session key. This
is the modern standard approach (same family as Signal Protocol, SSH,
TLS 1.3) -- it doesn't encrypt data directly, and doesn't need to: it
exists to set up a session key once, cheaply, which a fast symmetric
cipher then uses per message. This matters a lot on a ~15-20 char/sec
channel, where a raw RSA-style ciphertext (190+ bytes minimum regardless
of plaintext length) would dominate transmission time; ChaCha20-Poly1305's
fixed 28-byte overhead (12-byte nonce + 16-byte auth tag) per message is
the only recurring crypto cost here.

The ECDH shared secret is NOT used directly as the cipher key -- it's
passed through HKDF-SHA256 first (standard practice: raw ECDH output isn't
uniformly random enough to use directly as a symmetric key).

Caveat for demo use: there is no public-key verification here -- no PKI,
no out-of-band fingerprint check like Signal's safety numbers. A key
exchange carried out live over an open acoustic channel is, in principle,
vulnerable to a man-in-the-middle who intercepts and substitutes their own
public key. This is fine for demonstrating encrypted-in-transit messaging;
it is not a claim of protection against an active attacker without adding
that verification step separately.
"""
from __future__ import annotations

import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

NONCE_LEN = 12
TAG_LEN = 16  # produced by ChaCha20Poly1305 as part of its ciphertext output


def generate_keypair() -> tuple[X25519PrivateKey, X25519PublicKey]:
    private_key = X25519PrivateKey.generate()
    return private_key, private_key.public_key()


def serialize_public_key(pub: X25519PublicKey) -> bytes:
    return pub.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)


def load_public_key(data: bytes) -> X25519PublicKey:
    return X25519PublicKey.from_public_bytes(data)


def serialize_private_key(priv: X25519PrivateKey) -> bytes:
    return priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )


def load_private_key(data: bytes) -> X25519PrivateKey:
    return X25519PrivateKey.from_private_bytes(data)


def derive_session_key(my_private: X25519PrivateKey, their_public: X25519PublicKey,
                        info: bytes = b"textovervoice-session") -> bytes:
    """ECDH shared secret -> HKDF-SHA256 -> 32-byte symmetric key. Both
    sides land on the identical key: Alice with (her private, Bob's
    public), Bob with (his private, Alice's public)."""
    shared_secret = my_private.exchange(their_public)
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info).derive(shared_secret)


def encrypt(session_key: bytes, plaintext: bytes) -> bytes:
    nonce = os.urandom(NONCE_LEN)
    ciphertext = ChaCha20Poly1305(session_key).encrypt(nonce, plaintext, None)
    return nonce + ciphertext  # ciphertext already includes the 16-byte auth tag


def decrypt(session_key: bytes, blob: bytes) -> bytes | None:
    """Returns the plaintext, or None if authentication failed (tampered
    ciphertext, or simply the wrong key -- ChaCha20-Poly1305 can't tell
    those apart, which is the correct behavior: it shouldn't leak which
    one happened)."""
    if len(blob) < NONCE_LEN + TAG_LEN:
        return None
    nonce, ciphertext = blob[:NONCE_LEN], blob[NONCE_LEN:]
    try:
        return ChaCha20Poly1305(session_key).decrypt(nonce, ciphertext, None)
    except InvalidTag:
        return None
