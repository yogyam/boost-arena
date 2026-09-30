"""Checks that a sealed model can only be opened with the right key."""

import os

import pytest

from boost_arena.sealed import SealedFileError, generate_key_pair, is_sealed, open_sealed, seal


def test_round_trip():
    private, public = generate_key_pair()
    data = os.urandom(50_000)
    sealed = seal(data, public)
    assert is_sealed(sealed)
    assert data not in sealed
    assert open_sealed(sealed, private) == data


def test_every_seal_is_different():
    _, public = generate_key_pair()
    assert seal(b"same model", public) != seal(b"same model", public)


def test_the_wrong_key_does_not_open_it():
    private, public = generate_key_pair()
    other_private, _ = generate_key_pair()
    sealed = seal(b"secret model", public)
    with pytest.raises(SealedFileError, match="not sealed for this key"):
        open_sealed(sealed, other_private)


def test_a_changed_file_does_not_open():
    private, public = generate_key_pair()
    sealed = bytearray(seal(b"secret model", public))
    sealed[-1] ^= 1
    with pytest.raises(SealedFileError):
        open_sealed(bytes(sealed), private)
    sealed = bytearray(seal(b"secret model", public))
    sealed[len(b"BOOSTARENA-SEALED-1\n") + 3] ^= 1   # Inside the header
    with pytest.raises(SealedFileError):
        open_sealed(bytes(sealed), private)


def test_other_files_are_refused():
    private, _ = generate_key_pair()
    assert not is_sealed(b"just some bytes")
    with pytest.raises(SealedFileError, match="not a sealed model"):
        open_sealed(b"just some bytes", private)


def test_bad_keys_are_refused():
    with pytest.raises(SealedFileError):
        seal(b"x", "not a key")
    with pytest.raises(SealedFileError):
        open_sealed(seal(b"x", generate_key_pair()[1]), "bm90IGEga2V5")
