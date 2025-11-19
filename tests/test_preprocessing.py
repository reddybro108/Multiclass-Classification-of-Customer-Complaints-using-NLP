import pytest
import pandas as pd
from app.preprocessing import preprocess_text

def test_preprocess_text_standard():
    """Tests standard preprocessing: lowercasing, punctuation removal, stopword removal, and lemmatization."""
    text = "I was running and saw the cats, then I got charged an EXTRA fee! This is #unacceptable."
    expected = "running saw cat got charged extra fee unacceptable"
    assert preprocess_text(text, min_token_length=3) == expected

def test_preprocess_text_empty_string():
    """Tests that an empty string input results in an empty string output."""
    text = ""
    expected = ""
    assert preprocess_text(text, min_token_length=3) == expected

def test_preprocess_text_with_numbers():
    """Tests that numbers are removed."""
    text = "I paid 100 dollars for this service on 01/01/2025."
    expected = "paid dollar service"
    assert preprocess_text(text, min_token_length=3) == expected

def test_preprocess_text_only_stopwords():
    """Tests a string containing only stopwords."""
    text = "this is a and or the"
    expected = ""
    assert preprocess_text(text, min_token_length=3) == expected

def test_preprocess_text_nan_input():
    """Tests that a NaN input (common in pandas) results in an empty string."""
    text = pd.NA
    expected = ""
    assert preprocess_text(text, min_token_length=3) == expected

def test_preprocess_text_no_changes_needed():
    """Tests text that should remain unchanged after processing."""
    text = "word another"
    expected = "word another"
    assert preprocess_text(text, min_token_length=3) == expected

def test_preprocess_text_token_length():
    """Tests the minimum token length filtering."""
    text = "a be go in it to do so"
    expected = ""
    assert preprocess_text(text, min_token_length=3) == expected
    text = "cat dog car"
    expected = "cat dog car"
    assert preprocess_text(text, min_token_length=3) == expected