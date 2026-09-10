import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from textovervoice import crypto


def test_ecdh_both_sides_derive_same_session_key():
    alice_priv, alice_pub = crypto.generate_keypair()
    bob_priv, bob_pub = crypto.generate_keypair()

    alice_key = crypto.derive_session_key(alice_priv, bob_pub)
    bob_key = crypto.derive_session_key(bob_priv, alice_pub)
    assert alice_key == bob_key
    assert len(alice_key) == 32


def test_different_keypairs_derive_different_session_keys():
    alice_priv, alice_pub = crypto.generate_keypair()
    bob_priv, bob_pub = crypto.generate_keypair()
    eve_priv, eve_pub = crypto.generate_keypair()

    alice_bob_key = crypto.derive_session_key(alice_priv, bob_pub)
    alice_eve_key = crypto.derive_session_key(alice_priv, eve_pub)
    assert alice_bob_key != alice_eve_key


def test_encrypt_decrypt_roundtrip():
    priv, pub = crypto.generate_keypair()
    key = crypto.derive_session_key(priv, pub)  # self-exchange is fine for this test
    plaintext = "Hello, encrypted world! 你好".encode("utf-8")

    blob = crypto.encrypt(key, plaintext)
    assert crypto.decrypt(key, blob) == plaintext


def test_tampered_ciphertext_fails_auth():
    priv, pub = crypto.generate_keypair()
    key = crypto.derive_session_key(priv, pub)
    blob = bytearray(crypto.encrypt(key, b"secret message"))
    blob[-1] ^= 0xFF
    assert crypto.decrypt(key, bytes(blob)) is None


def test_wrong_key_fails_to_decrypt():
    alice_priv, alice_pub = crypto.generate_keypair()
    bob_priv, bob_pub = crypto.generate_keypair()
    eve_priv, eve_pub = crypto.generate_keypair()

    alice_key = crypto.derive_session_key(alice_priv, bob_pub)
    eve_key = crypto.derive_session_key(eve_priv, alice_pub)  # not the shared key

    blob = crypto.encrypt(alice_key, b"for bob's eyes only")
    assert crypto.decrypt(eve_key, blob) is None


def test_public_key_serialization_roundtrip():
    _, pub = crypto.generate_keypair()
    data = crypto.serialize_public_key(pub)
    assert len(data) == 32  # X25519 public keys are 32 raw bytes
    restored = crypto.load_public_key(data)
    assert crypto.serialize_public_key(restored) == data


def test_private_key_serialization_roundtrip():
    priv, pub = crypto.generate_keypair()
    data = crypto.serialize_private_key(priv)
    assert len(data) == 32
    restored = crypto.load_private_key(data)
    # verify it's functionally the same key via a session-key derivation
    other_priv, other_pub = crypto.generate_keypair()
    k1 = crypto.derive_session_key(priv, other_pub)
    k2 = crypto.derive_session_key(restored, other_pub)
    assert k1 == k2


def test_each_encryption_uses_a_fresh_nonce():
    priv, pub = crypto.generate_keypair()
    key = crypto.derive_session_key(priv, pub)
    blob1 = crypto.encrypt(key, b"same message")
    blob2 = crypto.encrypt(key, b"same message")
    assert blob1 != blob2  # different nonce each time, even for identical plaintext
