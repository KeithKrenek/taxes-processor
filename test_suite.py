# python/tests/test_pipeline.py - Main Pipeline Tests
"""
Comprehensive test suite for the Financial Transaction Processing Pipeline.
Demonstrates testing best practices including unit tests, integration tests,
mocking, parameterized tests, and performance testing.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path
import json

# Import modules to test
from processor.pipeline import (
    FinancialPipeline, 
    TransactionRecord, 
    ProcessingConfig,
    DataLoader,
    MLCategorizer,
    DuplicateDetector,
    FinancialVisualizer
)


class TestTransactionRecord:
    """Test cases for TransactionRecord data class."""
    
    def test_transaction_record_creation_valid(self):
        """Test creating a valid transaction record."""
        transaction = TransactionRecord(
            date=datetime(2024, 1, 15),
            amount=125.50,
            description="GROCERY STORE PURCHASE",
            account="checking",
            category="Food & Dining",
            confidence=0.85
        )
        
        assert transaction.date == datetime(2024, 1, 15)
        assert transaction.amount == 125.50
        assert transaction.description == "GROCERY STORE PURCHASE"
        assert transaction.category == "Food & Dining"
        assert transaction.confidence == 0.85
        assert not transaction.is_duplicate
    
    def test_transaction_record_validation(self):
        """Test transaction record validation."""
        # Test invalid confidence
        with pytest.raises(ValueError):
            TransactionRecord(
                date=datetime(2024, 1, 15),
                amount=125.50,
                description="Test",
                account="checking",
                confidence=1.5  # Invalid confidence > 1
            )
    
    @pytest.mark.parametrize("amount,expected_abs", [
        (100.0, 100.0),
        (-50.25, 50.25),
        (0.0, 0.0),
        (999.99, 999.99)
    ])
    def test_amount_handling(self, amount, expected_abs):
        """Test various amount values."""
        transaction = TransactionRecord(
            date=datetime(2024, 1, 15),
            amount=amount,
            description="Test transaction",
            account="test"
        )
        assert abs(transaction.amount) == expected_abs


class TestProcessingConfig:
    """Test cases for ProcessingConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ProcessingConfig()
        
        assert config.ml_confidence_threshold == 0.8
        assert config.enable_fuzzy_matching is True
        assert config.temporal_window_days == 5
        assert config.amount_tolerance == 0.01
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_data = {
            'ml_confidence_threshold': 0.9,
            'fuzzy_threshold': 90,
            'temporal_window_days': 3
        }
        
        config = ProcessingConfig(**config_data)
        
        assert config.ml_confidence_threshold == 0.9
        assert config.fuzzy_threshold == 90
        assert config.temporal_window_days == 3
    
    def test_config_from_yaml(self, tmp_path):
        """Test loading config from YAML file."""
        config_content = """
        ml_confidence_threshold: 0.85
        enable_fuzzy_matching: false
        temporal_window_days: 7
        """
        
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        
        config = ProcessingConfig.from_yaml(str(config_file))
        
        assert config.ml_confidence_threshold == 0.85
        assert config.enable_fuzzy_matching is False
        assert config.temporal_window_days == 7


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    @pytest.fixture
    def sample_excel_data(self):
        """Create sample Excel data for testing."""
        data = {
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Amount': [100.50, -25.75, 200.00],
            'Description': ['GROCERY STORE', 'ATM WITHDRAWAL', 'SALARY DEPOSIT'],
            'Account': ['Checking', 'Checking', 'Checking']
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def mock_excel_file(self, tmp_path, sample_excel_data):
        """Create a mock Excel file for testing."""
        file_path = tmp_path / "test_transactions.xlsx"
        
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            sample_excel_data.to_excel(writer, sheet_name='Checking', index=False)
            sample_excel_data.to_excel(writer, sheet_name='Savings', index=False)
            
            # Add category sheets that should be ignored
            categories_df = pd.DataFrame({'Category': ['Food', 'Transport']})
            categories_df.to_excel(writer, sheet_name='TransactionCategories', index=False)
        
        return file_path
    
    def test_load_workbook_success(self, mock_excel_file):
        """Test successful workbook loading."""
        config = ProcessingConfig()
        loader = DataLoader(config)
        
        transactions = loader.load_workbook(str(mock_excel_file))
        
        assert len(transactions) == 6  # 3 transactions × 2 sheets
        assert all(isinstance(t, TransactionRecord) for t in transactions)
        assert transactions[0].description == "GROCERY STORE"
    
    def test_load_workbook_file_not_found(self):
        """Test handling of missing file."""
        config = ProcessingConfig()
        loader = DataLoader(config)
        
        with pytest.raises(Exception):
            loader.load_workbook("nonexistent_file.xlsx")
    
    @pytest.mark.parametrize("column_name,expected_mapping", [
        ("posting date", "date"),
        ("transaction amount", "amount"), 
        ("memo", "description"),
        ("trans date", "date")
    ])
    def test_column_mapping(self, column_name, expected_mapping):
        """Test column name mapping functionality."""
        config = ProcessingConfig()
        loader = DataLoader(config)
        
        columns = [column_name, "other_column"]
        mapping = loader._map_columns(columns)
        
        assert expected_mapping in mapping.values()
    
    def test_clean_description(self):
        """Test description cleaning functionality."""
        config = ProcessingConfig()
        loader = DataLoader(config)
        
        dirty_description = "  PURCHASE   AT  STORE\x00  "
        clean_description = loader._clean_description(dirty_description)
        
        assert clean_description == "AT STORE"
        assert "\x00" not in clean_description


class TestMLCategorizer:
    """Test cases for ML Categorizer."""
    
    @pytest.fixture
    def sample_transactions(self):
        """Create sample transactions for testing."""
        return [
            TransactionRecord(
                date=datetime(2024, 1, 1),
                amount=25.50,
                description="STARBUCKS COFFEE",
                account="checking"
            ),
            TransactionRecord(
                date=datetime(2024, 1, 2),
                amount=45.00,
                description="SHELL GAS STATION",
                account="checking"
            ),
            TransactionRecord(
                date=datetime(2024, 1, 3),
                amount=1250.00,
                description="RENT PAYMENT",
                account="checking"
            )
        ]
    
    def test_categorizer_initialization(self):
        """Test categorizer initialization."""
        config = ProcessingConfig()
        categorizer = MLCategorizer(config)
        
        assert categorizer.config == config
        assert "Food & Dining" in categorizer.category_keywords
        assert not categorizer.trained
    
    def test_rule_based_categorization(self, sample_transactions):
        """Test rule-based categorization."""
        config = ProcessingConfig()
        categorizer = MLCategorizer(config)
        
        # Test coffee shop categorization
        category, confidence = categorizer._rule_based_categorization(sample_transactions[0])
        assert category == "Food & Dining"
        assert confidence > 0.5
        
        # Test gas station categorization
        category, confidence = categorizer._rule_based_categorization(sample_transactions[1])
        assert category == "Transportation"
        assert confidence > 0.5
    
    def test_feature_extraction(self, sample_transactions):
        """Test feature extraction for ML."""
        config = ProcessingConfig()
        categorizer = MLCategorizer(config)
        
        features = categorizer.extract_features(sample_transactions)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == 3
        assert 'length' in features.columns
        assert 'amount' in features.columns
        assert 'day_of_week' in features.columns
    
    def test_categorize_transactions(self, sample_transactions):
        """Test full transaction categorization."""
        config = ProcessingConfig()
        categorizer = MLCategorizer(config)
        
        categorized = categorizer.categorize_transactions(sample_transactions)
        
        assert len(categorized) == 3
        assert all(t.category is not None for t in categorized)
        assert all(0 <= t.confidence <= 1 for t in categorized)
    
    @patch('processor.pipeline.cross_val_score')
    @patch('processor.pipeline.RandomForestClassifier')
    def test_ml_training(self, mock_rf, mock_cv, sample_transactions):
        """Test ML model training with mocked dependencies."""
        config = ProcessingConfig()
        categorizer = MLCategorizer(config)
        
        # Mock cross-validation scores
        mock_cv.return_value = [0.85, 0.87, 0.89, 0.86, 0.88]
        
        labels = ["Food & Dining", "Transportation", "Housing"]
        categorizer.train(sample_transactions, labels)
        
        assert categorizer.trained
        mock_cv.assert_called_once()


class TestDuplicateDetector:
    """Test cases for Duplicate Detector."""
    
    @pytest.fixture
    def duplicate_transactions(self):
        """Create transactions with duplicates for testing."""
        base_date = datetime(2024, 1, 1)
        return [
            TransactionRecord(
                date=base_date,
                amount=100.00,
                description="GROCERY STORE PURCHASE",
                account="checking"
            ),
            TransactionRecord(
                date=base_date + timedelta(days=1),
                amount=100.00,
                description="GROCERY STORE PURCHASE",
                account="savings"  # Same transaction, different account
            ),
            TransactionRecord(
                date=base_date + timedelta(days=10),
                amount=50.00,
                description="DIFFERENT TRANSACTION",
                account="checking"
            )
        ]
    
    def test_duplicate_detection(self, duplicate_transactions):
        """Test duplicate detection functionality."""
        config = ProcessingConfig()
        detector = DuplicateDetector(config)
        
        processed = detector.detect_duplicates(duplicate_transactions)
        
        # First two transactions should be marked as duplicates
        assert processed[0].is_duplicate
        assert processed[1].is_duplicate
        assert not processed[2].is_duplicate
        
        # They should have the same duplicate group
        assert processed[0].duplicate_group == processed[1].duplicate_group
    
    def test_description_similarity(self):
        """Test description similarity calculation."""
        config = ProcessingConfig()
        detector = DuplicateDetector(config)
        
        # Test identical descriptions
        similarity = detector._calculate_description_similarity(
            "GROCERY STORE PURCHASE",
            "GROCERY STORE PURCHASE"
        )
        assert similarity > 0.9
        
        # Test similar descriptions
        similarity = detector._calculate_description_similarity(
            "GROCERY STORE PURCHASE",
            "GROCERY STORE BUY"
        )
        assert 0.5 < similarity < 0.9
        
        # Test different descriptions
        similarity = detector._calculate_description_similarity(
            "GROCERY STORE",
            "GAS STATION"
        )
        assert similarity < 0.5
    
    def test_temporal_window(self, duplicate_transactions):
        """Test temporal window for duplicate detection."""
        config = ProcessingConfig(temporal_window_days=2)
        detector = DuplicateDetector(config)
        
        # Modify one transaction to be outside temporal window
        duplicate_transactions[1].date = duplicate_transactions[0].date + timedelta(days=5)
        
        processed = detector.detect_duplicates(duplicate_transactions)
        
        # Should not be detected as duplicates due to temporal distance
        assert not processed[0].is_duplicate
        assert not processed[1].is_duplicate
    
    @pytest.mark.parametrize("amount_diff,should_be_duplicate", [
        (0.00, True),   # Exact match
        (0.01, False),  # Within tolerance (assuming default 0.01)
        (0.02, False),  # Outside tolerance
        (0.005, True),  # Within tolerance
    ])
    def test_amount_tolerance(self, amount_diff, should_be_duplicate):
        """Test amount tolerance in duplicate detection."""
        config = ProcessingConfig(amount_tolerance=0.01)
        detector = DuplicateDetector(config)
        
        base_date = datetime(2024, 1, 1)
        transactions = [
            TransactionRecord(
                date=base_date,
                amount=100.00,
                description="TEST TRANSACTION",
                account="checking"
            ),
            TransactionRecord(
                date=base_date,
                amount=100.00 + amount_diff,
                description="TEST TRANSACTION",
                account="savings"
            )
        ]
        
        processed = detector.detect_duplicates(transactions)
        
        if should_be_duplicate:
            assert processed[0].is_duplicate and processed[1].is_duplicate
        else:
            assert not processed[0].is_duplicate and not processed[1].is_duplicate


class TestFinancialVisualizer:
    """Test cases for Financial Visualizer."""
    
    @pytest.fixture
    def sample_transactions_for_viz(self):
        """Create sample transactions for visualization testing."""
        base_date = datetime(2024, 1, 1)
        return [
            TransactionRecord(
                date=base_date,
                amount=25.00,
                description="COFFEE SHOP",
                account="checking",
                category="Food & Dining",
                confidence=0.9
            ),
            TransactionRecord(
                date=base_date + timedelta(days=1),
                amount=100.00,
                description="GROCERY STORE",
                account="checking",
                category="Food & Dining",
                confidence=0.85
            ),
            TransactionRecord(
                date=base_date + timedelta(days=2),
                amount=50.00,
                description="GAS STATION",
                account="checking",
                category="Transportation",
                confidence=0.92
            )
        ]
    
    def test_transactions_to_dataframe(self, sample_transactions_for_viz):
        """Test conversion of transactions to DataFrame."""
        config = ProcessingConfig()
        visualizer = FinancialVisualizer(config)
        
        df = visualizer._transactions_to_dataframe(sample_transactions_for_viz)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'date' in df.columns
        assert 'amount' in df.columns
        assert 'category' in df.columns
    
    def test_monthly_data_preparation(self, sample_transactions_for_viz):
        """Test monthly data aggregation."""
        config = ProcessingConfig()
        visualizer = FinancialVisualizer(config)
        
        df = visualizer._transactions_to_dataframe(sample_transactions_for_viz)
        monthly_data = visualizer._prepare_monthly_data(df)
        
        assert isinstance(monthly_data, pd.DataFrame)
        assert 'amount' in monthly_data.columns
        assert 'transaction_count' in monthly_data.columns
    
    def test_duplicate_stats_calculation(self, sample_transactions_for_viz):
        """Test duplicate statistics calculation."""
        config = ProcessingConfig()
        visualizer = FinancialVisualizer(config)
        
        # Mark one transaction as duplicate
        sample_transactions_for_viz[0].is_duplicate = True
        sample_transactions_for_viz[0].duplicate_group = 1
        
        df = visualizer._transactions_to_dataframe(sample_transactions_for_viz)
        stats = visualizer._calculate_duplicate_stats(df)
        
        assert stats['Total Transactions'] == 3
        assert stats['Unique Transactions'] == 2
        assert stats['Duplicate Transactions'] == 1
        assert stats['Duplicate Groups'] == 1
    
    @patch('plotly.graph_objects.Figure.write_html')
    def test_dashboard_creation(self, mock_write_html, sample_transactions_for_viz):
        """Test dashboard creation with mocked file writing."""
        config = ProcessingConfig()
        visualizer = FinancialVisualizer(config)
        
        # Should not raise an exception
        fig = visualizer.create_comprehensive_dashboard(
            sample_transactions_for_viz,
            "test_dashboard.html"
        )
        
        mock_write_html.assert_called_once_with("test_dashboard.html")


class TestFinancialPipeline:
    """Integration tests for the main FinancialPipeline class."""
    
    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing."""
        return ProcessingConfig(
            ml_confidence_threshold=0.8,
            temporal_window_days=5,
            amount_tolerance=0.01
        )
    
    @pytest.fixture
    def mock_excel_file_complex(self, tmp_path):
        """Create a more complex Excel file for integration testing."""
        # Create main transaction data
        transaction_data = {
            'Posting Date': [
                '2024-01-01', '2024-01-02', '2024-01-02', '2024-01-05',
                '2024-01-10', '2024-01-15', '2024-01-20'
            ],
            'Amount': [25.50, -100.00, -100.00, 45.75, 1200.00, -50.25, -25.50],
            'Description': [
                'STARBUCKS COFFEE #123',
                'GROCERY STORE PURCHASE',
                'GROCERY STORE PURCHASE',  # Duplicate
                'SHELL GAS STATION',
                'SALARY DEPOSIT COMPANY',
                'AMAZON PURCHASE',
                'COFFEE SHOP DOWNTOWN'
            ]
        }
        
        file_path = tmp_path / "integration_test.xlsx"
        
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Main transactions
            df = pd.DataFrame(transaction_data)
            df.to_excel(writer, sheet_name='Checking', index=False)
            
            # Additional account
            df_savings = df.iloc[:3].copy()  # First 3 transactions
            df_savings.to_excel(writer, sheet_name='Savings', index=False)
            
            # Category reference sheets (should be ignored)
            categories = pd.DataFrame({
                'Category': ['Food & Dining', 'Transportation', 'Shopping']
            })
            categories.to_excel(writer, sheet_name='TransactionCategories', index=False)
        
        return file_path
    
    def test_full_pipeline_integration(self, sample_config, mock_excel_file_complex):
        """Test the complete pipeline end-to-end."""
        pipeline = FinancialPipeline()
        pipeline.config = sample_config
        
        results = pipeline.process_file(str(mock_excel_file_complex))
        
        # Verify results structure
        assert 'transactions' in results
        assert 'summary' in results
        assert 'processing_time' in results
        assert 'config' in results
        
        # Verify transactions were processed
        transactions = results['transactions']
        assert len(transactions) > 0
        assert all(isinstance(t, TransactionRecord) for t in transactions)
        
        # Verify categorization occurred
        categorized_count = sum(1 for t in transactions if t.category and t.category != 'Uncategorized')
        assert categorized_count > 0
        
        # Verify duplicate detection occurred
        duplicates = [t for t in transactions if t.is_duplicate]
        assert len(duplicates) >= 2  # We expect at least the intentional duplicates
        
        # Verify summary statistics
        summary = results['summary']
        assert summary['total_transactions'] == len(transactions)
        assert summary['duplicate_count'] == len(duplicates)
        assert 'categories' in summary
        assert 'confidence_stats' in summary
    
    def test_pipeline_error_handling(self, sample_config):
        """Test pipeline error handling with invalid file."""
        pipeline = FinancialPipeline()
        pipeline.config = sample_config
        
        with pytest.raises(Exception):
            pipeline.process_file("nonexistent_file.xlsx")
    
    def test_generate_reports(self, sample_config, mock_excel_file_complex, tmp_path):
        """Test report generation functionality."""
        pipeline = FinancialPipeline()
        pipeline.config = sample_config
        
        results = pipeline.process_file(str(mock_excel_file_complex))
        
        # Generate reports in temporary directory
        output_dir = tmp_path / "reports"
        pipeline.generate_reports(results, str(output_dir))
        
        # Verify output files were created
        assert (output_dir / "financial_dashboard.html").exists()
        assert (output_dir / "processed_transactions.xlsx").exists()
        assert (output_dir / "processing_report.txt").exists()
    
    @patch('processor.pipeline.FinancialVisualizer.create_comprehensive_dashboard')
    def test_visualization_integration(self, mock_dashboard, sample_config, mock_excel_file_complex):
        """Test visualization integration with mocked dashboard creation."""
        pipeline = FinancialPipeline()
        pipeline.config = sample_config
        
        results = pipeline.process_file(str(mock_excel_file_complex))
        pipeline.generate_reports(results)
        
        # Verify dashboard creation was called
        mock_dashboard.assert_called_once()


class TestPerformance:
    """Performance and stress tests."""
    
    def test_large_dataset_performance(self):
        """Test performance with large dataset."""
        # Create large dataset
        import time
        
        base_date = datetime(2024, 1, 1)
        transactions = []
        
        # Generate 10,000 transactions
        for i in range(10000):
            transactions.append(TransactionRecord(
                date=base_date + timedelta(days=i % 365),
                amount=float(i % 1000 + 10),
                description=f"Transaction {i % 100}",
                account="test"
            ))
        
        config = ProcessingConfig()
        detector = DuplicateDetector(config)
        
        start_time = time.time()
        processed = detector.detect_duplicates(transactions)
        processing_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert processing_time < 30.0  # 30 seconds max for 10k transactions
        assert len(processed) == 10000
    
    @pytest.mark.parametrize("transaction_count", [100, 1000, 5000])
    def test_scalability(self, transaction_count):
        """Test scalability with different dataset sizes."""
        base_date = datetime(2024, 1, 1)
        transactions = []
        
        for i in range(transaction_count):
            transactions.append(TransactionRecord(
                date=base_date + timedelta(days=i % 100),
                amount=float(i % 500 + 1),
                description=f"Test transaction {i}",
                account="test"
            ))
        
        config = ProcessingConfig()
        categorizer = MLCategorizer(config)
        
        import time
        start_time = time.time()
        categorized = categorizer.categorize_transactions(transactions)
        processing_time = time.time() - start_time
        
        assert len(categorized) == transaction_count
        # Processing time should scale reasonably
        assert processing_time < (transaction_count / 100)  # Max 1 second per 100 transactions


# Fixtures for test data
@pytest.fixture(scope="session")
def sample_test_data():
    """Session-scoped test data for reuse across tests."""
    return {
        'transactions': [
            {
                'date': '2024-01-01',
                'amount': 25.50,
                'description': 'COFFEE SHOP',
                'category': 'Food & Dining'
            },
            {
                'date': '2024-01-02', 
                'amount': -100.00,
                'description': 'ATM WITHDRAWAL',
                'category': 'Cash'
            }
        ]
    }


# Custom test markers
pytestmark = [
    pytest.mark.unit,  # Mark all tests in this file as unit tests
]


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
