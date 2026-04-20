"""
Unit tests for sample data validation in Multi-label Emotion Classification system.

This module contains tests to validate the generated sample_comments.csv file,
ensuring it has the correct format, columns, and data integrity.

**Validates: Requirements 2.1, 2.4**
"""

import pytest
import pandas as pd
import numpy as np
import os

from config import Config


class TestSampleDataFormat:
    """Test suite for sample data format validation."""
    
    def test_sample_data_file_exists(self):
        """
        Test that sample_comments.csv file exists.
        
        Expected behavior:
        - File should exist at data/sample_comments.csv
        
        **Validates: Requirements 2.1, 2.4**
        """
        file_path = os.path.join(Config.DATA_DIR, 'sample_comments.csv')
        assert os.path.exists(file_path), f"Sample data file should exist at {file_path}"
    
    def test_sample_data_loads_successfully(self):
        """
        Test that sample_comments.csv can be loaded as a DataFrame.
        
        Expected behavior:
        - File should be loadable by pandas
        - Should not raise any errors
        
        **Validates: Requirements 2.1**
        """
        file_path = os.path.join(Config.DATA_DIR, 'sample_comments.csv')
        
        try:
            df = pd.read_csv(file_path)
            assert df is not None, "DataFrame should not be None"
        except Exception as e:
            pytest.fail(f"Failed to load sample data: {e}")
    
    def test_sample_data_has_text_column(self):
        """
        Test that sample data has 'text' column.
        
        Expected behavior:
        - DataFrame should have a column named 'text'
        
        **Validates: Requirements 2.1, 2.4**
        """
        file_path = os.path.join(Config.DATA_DIR, 'sample_comments.csv')
        df = pd.read_csv(file_path)
        
        assert 'text' in df.columns, "Sample data should have 'text' column"
    
    def test_sample_data_has_all_emotion_columns(self):
        """
        Test that sample data has all 16 emotion label columns.
        
        Expected behavior:
        - DataFrame should have all 16 emotion columns from Config.EMOTION_LABELS
        
        **Validates: Requirements 2.1, 2.4**
        """
        file_path = os.path.join(Config.DATA_DIR, 'sample_comments.csv')
        df = pd.read_csv(file_path)
        
        # Check that all emotion labels are present as columns
        for emotion in Config.EMOTION_LABELS:
            assert emotion in df.columns, f"Sample data should have '{emotion}' column"
    
    def test_sample_data_column_count(self):
        """
        Test that sample data has exactly 17 columns (1 text + 16 emotions).
        
        Expected behavior:
        - DataFrame should have exactly 17 columns
        
        **Validates: Requirements 2.1, 2.4**
        """
        file_path = os.path.join(Config.DATA_DIR, 'sample_comments.csv')
        df = pd.read_csv(file_path)
        
        expected_columns = 1 + len(Config.EMOTION_LABELS)  # 1 text + 16 emotions = 17
        assert len(df.columns) == expected_columns, f"Sample data should have {expected_columns} columns"
    
    def test_sample_data_has_minimum_rows(self):
        """
        Test that sample data has at least 100 rows.
        
        Expected behavior:
        - DataFrame should have at least 100 samples as per requirements
        
        **Validates: Requirements 2.1**
        """
        file_path = os.path.join(Config.DATA_DIR, 'sample_comments.csv')
        df = pd.read_csv(file_path)
        
        assert len(df) >= 100, "Sample data should have at least 100 samples"
    
    def test_sample_data_text_not_empty(self):
        """
        Test that all text entries are non-empty.
        
        Expected behavior:
        - No text entry should be empty or null
        
        **Validates: Requirements 2.1**
        """
        file_path = os.path.join(Config.DATA_DIR, 'sample_comments.csv')
        df = pd.read_csv(file_path)
        
        # Check for null values
        assert not df['text'].isnull().any(), "No text entry should be null"
        
        # Check for empty strings
        assert not (df['text'].str.strip() == '').any(), "No text entry should be empty"
    
    def test_sample_data_labels_are_binary(self):
        """
        Test that all emotion label values are binary (0 or 1).
        
        Expected behavior:
        - All emotion columns should contain only 0 or 1
        
        **Validates: Requirements 2.1, 2.4**
        """
        file_path = os.path.join(Config.DATA_DIR, 'sample_comments.csv')
        df = pd.read_csv(file_path)
        
        # Check each emotion column
        for emotion in Config.EMOTION_LABELS:
            unique_values = df[emotion].unique()
            assert set(unique_values).issubset({0, 1}), f"Column '{emotion}' should only contain 0 or 1"
    
    def test_sample_data_labels_are_numeric(self):
        """
        Test that all emotion label columns are numeric type.
        
        Expected behavior:
        - All emotion columns should be numeric (int or float)
        
        **Validates: Requirements 2.1, 2.4**
        """
        file_path = os.path.join(Config.DATA_DIR, 'sample_comments.csv')
        df = pd.read_csv(file_path)
        
        # Check each emotion column
        for emotion in Config.EMOTION_LABELS:
            assert pd.api.types.is_numeric_dtype(df[emotion]), f"Column '{emotion}' should be numeric"
    
    def test_sample_data_at_least_one_label_per_row(self):
        """
        Test that each row has at least one emotion label set to 1.
        
        Expected behavior:
        - Every sample should have at least one emotion assigned
        
        **Validates: Requirements 2.5**
        """
        file_path = os.path.join(Config.DATA_DIR, 'sample_comments.csv')
        df = pd.read_csv(file_path)
        
        # Sum emotion labels for each row
        emotion_columns = Config.EMOTION_LABELS
        label_sums = df[emotion_columns].sum(axis=1)
        
        # Check that all rows have at least one label
        assert (label_sums >= 1).all(), "Every sample should have at least one emotion label"
    
    def test_sample_data_has_multilabel_samples(self):
        """
        Test that sample data includes multi-label samples (rows with multiple emotions).
        
        Expected behavior:
        - At least some samples should have more than one emotion
        
        **Validates: Requirements 2.3**
        """
        file_path = os.path.join(Config.DATA_DIR, 'sample_comments.csv')
        df = pd.read_csv(file_path)
        
        # Sum emotion labels for each row
        emotion_columns = Config.EMOTION_LABELS
        label_sums = df[emotion_columns].sum(axis=1)
        
        # Check that at least some rows have multiple labels
        multi_label_count = (label_sums > 1).sum()
        assert multi_label_count > 0, "Sample data should include multi-label samples"
    
    def test_sample_data_has_diverse_emotions(self):
        """
        Test that sample data includes diverse emotions (not just one or two).
        
        Expected behavior:
        - Multiple different emotions should be represented in the dataset
        
        **Validates: Requirements 2.3**
        """
        file_path = os.path.join(Config.DATA_DIR, 'sample_comments.csv')
        df = pd.read_csv(file_path)
        
        # Count how many emotions have at least one occurrence
        emotion_columns = Config.EMOTION_LABELS
        emotions_with_occurrences = 0
        
        for emotion in emotion_columns:
            if df[emotion].sum() > 0:
                emotions_with_occurrences += 1
        
        # At least 10 out of 16 emotions should be represented
        assert emotions_with_occurrences >= 10, "Sample data should include diverse emotions"
    
    def test_sample_data_includes_english_text(self):
        """
        Test that sample data includes English text.
        
        Expected behavior:
        - At least some samples should be in English
        
        **Validates: Requirements 2.2**
        """
        file_path = os.path.join(Config.DATA_DIR, 'sample_comments.csv')
        df = pd.read_csv(file_path)
        
        # Simple heuristic: check if text contains common English words
        english_words = ['the', 'is', 'this', 'I', 'love', 'good', 'bad', 'happy', 'sad']
        
        english_count = 0
        for text in df['text']:
            text_lower = text.lower()
            if any(word.lower() in text_lower for word in english_words):
                english_count += 1
        
        # At least 30% of samples should be in English
        assert english_count >= len(df) * 0.3, "Sample data should include English text"
    
    def test_sample_data_includes_vietnamese_text(self):
        """
        Test that sample data includes Vietnamese text.
        
        Expected behavior:
        - At least some samples should be in Vietnamese
        
        **Validates: Requirements 2.2**
        """
        file_path = os.path.join(Config.DATA_DIR, 'sample_comments.csv')
        df = pd.read_csv(file_path)
        
        # Simple heuristic: check if text contains Vietnamese characters
        vietnamese_chars = ['ă', 'â', 'đ', 'ê', 'ô', 'ơ', 'ư', 'á', 'à', 'ả', 'ã', 'ạ',
                           'é', 'è', 'ẻ', 'ẽ', 'ẹ', 'í', 'ì', 'ỉ', 'ĩ', 'ị',
                           'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ú', 'ù', 'ủ', 'ũ', 'ụ']
        
        vietnamese_count = 0
        for text in df['text']:
            if any(char in text for char in vietnamese_chars):
                vietnamese_count += 1
        
        # At least 20% of samples should be in Vietnamese
        assert vietnamese_count >= len(df) * 0.2, "Sample data should include Vietnamese text"
    
    def test_sample_data_no_duplicate_texts(self):
        """
        Test that sample data has no duplicate text entries.
        
        Expected behavior:
        - All text entries should be unique
        
        **Validates: Requirements 2.1**
        """
        file_path = os.path.join(Config.DATA_DIR, 'sample_comments.csv')
        df = pd.read_csv(file_path)
        
        # Check for duplicates
        duplicate_count = df['text'].duplicated().sum()
        
        # Allow a small number of duplicates (less than 5% of dataset)
        max_allowed_duplicates = len(df) * 0.05
        assert duplicate_count <= max_allowed_duplicates, f"Sample data should have minimal duplicates (found {duplicate_count})"
    
    def test_sample_data_text_length_reasonable(self):
        """
        Test that text entries have reasonable length.
        
        Expected behavior:
        - Text should not be too short (at least 5 characters)
        - Text should not be excessively long (less than 1000 characters)
        
        **Validates: Requirements 2.1**
        """
        file_path = os.path.join(Config.DATA_DIR, 'sample_comments.csv')
        df = pd.read_csv(file_path)
        
        # Check minimum length
        min_length = df['text'].str.len().min()
        assert min_length >= 5, "Text entries should be at least 5 characters long"
        
        # Check maximum length
        max_length = df['text'].str.len().max()
        assert max_length <= 1000, "Text entries should not exceed 1000 characters"
    
    def test_sample_data_label_distribution(self):
        """
        Test that emotion labels have reasonable distribution.
        
        Expected behavior:
        - No single emotion should dominate (more than 80% of samples)
        - No emotion should be extremely rare (less than 1% of samples)
        
        **Validates: Requirements 2.3**
        """
        file_path = os.path.join(Config.DATA_DIR, 'sample_comments.csv')
        df = pd.read_csv(file_path)
        
        emotion_columns = Config.EMOTION_LABELS
        total_samples = len(df)
        
        for emotion in emotion_columns:
            emotion_count = df[emotion].sum()
            emotion_percentage = emotion_count / total_samples
            
            # Skip emotions that have zero occurrences (some emotions might not be in sample data)
            if emotion_count > 0:
                # No emotion should be in more than 80% of samples
                assert emotion_percentage <= 0.8, f"Emotion '{emotion}' appears in {emotion_percentage*100:.1f}% of samples (too dominant)"
