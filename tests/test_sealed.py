"""Checks that a sealed model can only be opened with the right key, for the submission it was sealed for."""

import os

import pytest

from boost_arena.sealed import MAGIC, SealedFileError, generate_key_pair, is_sealed, open_sealed, seal

BOT = ("my-bot", "someone")


def test_round_trip():
    private, public = generate_key_pair()
    data = os.urandom(50_000)
    sealed = seal(data, public, *BOT)
    assert is_sealed(sealed)
    assert data not in sealed
    assert open_sealed(sealed, private, *BOT) == data


def test_every_seal_is_different():
    _, public = generate_key_pair()
    assert seal(b"same model", public, *BOT) != seal(b"same model", public, *BOT)


def test_the_wrong_key_does_not_open_it():
    private, public = generate_key_pair()
    other_private, _ = generate_key_pair()
    sealed = seal(b"secret model", public, *BOT)
    with pytest.raises(SealedFileError, match="not sealed for this key"):
        open_sealed(sealed, other_private, *BOT)


def test_a_file_sealed_for_one_submission_does_not_open_for_another():
    """Copying someone's sealed file into your own submission would claim their score."""
    private, public = generate_key_pair()
    sealed = seal(b"secret model", public, "their-bot", "them")
    assert open_sealed(sealed, private, "their-bot", "them") == b"secret model"
    for slug, github in (("my-bot", "them"), ("their-bot", "me"), ("their-bo", "tthem"), ("their-bot\0", "them")):
        with pytest.raises(SealedFileError):
            open_sealed(sealed, private, slug, github)
    with pytest.raises(SealedFileError, match="bound"):
        seal(b"x", public, "", "someone")


def test_a_changed_file_does_not_open():
    private, public = generate_key_pair()
    sealed = bytearray(seal(b"secret model", public, *BOT))
    sealed[-1] ^= 1
    with pytest.raises(SealedFileError):
        open_sealed(bytes(sealed), private, *BOT)
    sealed = bytearray(seal(b"secret model", public, *BOT))
    sealed[len(MAGIC) + 3] ^= 1  # Inside the header
    with pytest.raises(SealedFileError):
        open_sealed(bytes(sealed), private, *BOT)


def test_other_files_are_refused():
    private, _ = generate_key_pair()
    assert not is_sealed(b"just some bytes")
    assert not is_sealed(b"BOOSTARENA-SEALED-1\n" + b"\0" * 100)  # The old format is not accepted
    with pytest.raises(SealedFileError, match="not a sealed model"):
        open_sealed(b"just some bytes", private, *BOT)


def test_bad_keys_are_refused():
    with pytest.raises(SealedFileError):
        seal(b"x", "not a key", *BOT)
    with pytest.raises(SealedFileError):
        open_sealed(seal(b"x", generate_key_pair()[1], *BOT), "bm90IGEga2V5", *BOT)
