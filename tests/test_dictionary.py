import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from textovervoice import dictionary
from textovervoice.charset import encode_text as charset_encode_text


def roundtrip(text: str) -> str:
    return dictionary.decode_codes(dictionary.encode_text(text))


def test_dictionary_word_roundtrips():
    assert roundtrip("people") == "people"


def test_title_case_roundtrips():
    assert roundtrip("People are here") == "People are here"


def test_all_caps_falls_back_to_literal_not_dictionary():
    text = "PEOPLE"
    codes_out = dictionary.encode_text(text)
    # no DICT flag emitted -- should be plain per-character codes
    from textovervoice import codes as C
    assert C.DICT_LOWER not in codes_out
    assert C.DICT_TITLE not in codes_out
    assert roundtrip(text) == text


def test_mixed_sentence_roundtrips():
    text = "The government should provide information about something important."
    assert roundtrip(text) == text


def test_non_dictionary_word_falls_back_unchanged():
    text = "xyzzyqux"  # not a real word, won't be in the dictionary
    assert roundtrip(text) == text


def test_punctuation_and_spacing_preserved():
    text = "Hello, world! Is this really working? Yes -- because it should."
    assert roundtrip(text) == text


def test_dictionary_hit_is_shorter_than_plain_encoding_for_long_common_words():
    text = "government information international"
    dict_len = len(dictionary.encode_text(text))
    plain_len = len(charset_encode_text(text))
    assert dict_len < plain_len


def test_short_words_dont_get_dictionary_entries():
    """Words <=4 letters can't win (a hit costs 4 codes: flag+3 letters),
    so the generator should never have assigned them entries -- verifies
    the build_dictionary.py filtering, not just this module's logic."""
    table = dictionary._load("en")
    assert all(len(word) >= 5 for word in table)


def test_empty_and_single_char_no_crash():
    assert roundtrip("") == ""
    assert roundtrip("x") == "x"


def test_cjk_text_unaffected_by_dictionary_layer():
    text = "你好，世界！"
    assert roundtrip(text) == text
