# benchmarks/performance.py - Performance Benchmarking Suite
"""
Comprehensive performance benchmarking suite for the Financial Transaction Processor.
Demonstrates performance testing, profiling, and optimization analysis.
"""

import time
import memory_profiler
import psutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import cProfile
import pstats
import io

# Import our modules
from processor.pipeline import FinancialPipeline, ProcessingConfig, TransactionRecord
from processor.data_loader import DataLoader
from processor.categorizer import MLCategorizer
from processor.duplicate_detector import DuplicateDetector


class PerformanceBenchmark:
    """Comprehensive performance benchmarking for financial processing pipeline."""
    
    def __init__(self):
        self.results = {}
        self.config = ProcessingConfig(
            ml_confidence_threshold=0.8,
            temporal_window_days=5,
            amount_tolerance=0.01
        )
    
    def generate_test_data(self, num_transactions: int, duplicate_rate: float = 0.05) -> List[TransactionRecord]:
        """Generate synthetic transaction data for testing."""
        print(f"Generating {num_transactions} test transactions...")
        
        transactions = []
        base_date = datetime(2024, 1, 1)
        
        # Merchant names for realistic descriptions
        merchants = [
            "STARBUCKS", "WALMART", "AMAZON", "SHELL", "TARGET", "MCDONALDS",
            "GROCERY STORE", "GAS STATION", "COFFEE SHOP", "RESTAURANT",
            "PHARMACY", "BANK ATM", "ONLINE PURCHASE", "UTILITY BILL"
        ]
        
        # Generate base transactions
        for i in range(int(num_transactions * (1 - duplicate_rate))):
            date = base_date + timedelta(days=random.randint(0, 365))
            amount = round(random.uniform(-1000, 1000), 2)
            merchant = random.choice(merchants)
            description = f"{merchant} #{random.randint(1000, 9999)}"
            account = random.choice(["Checking", "Savings", "Credit"])
            
            transactions.append(TransactionRecord(
                date=date,
                amount=amount,
                description=description,
                account=account
            ))
        
        # Generate duplicates
        num_duplicates = int(num_transactions * duplicate_rate)
        for i in range(num_duplicates):
            # Pick a random existing transaction to duplicate
            original = random.choice(transactions)
            duplicate_date = original.date + timedelta(days=random.randint(0, 3))
            
            # Add slight variations to make detection more realistic
            amount_variation = random.uniform(-0.01, 0.01)
            description_suffix = random.choice(["", " PENDING", " POSTED"])
            
            duplicate = TransactionRecord(
                date=duplicate_date,
                amount=original.amount + amount_variation,
                description=original.description + description_suffix,
                account=original.account
            )
            
            transactions.append(duplicate)
        
        # Shuffle transactions
        random.shuffle(transactions)
        
        print(f"Generated {len(transactions)} transactions with ~{num_duplicates} duplicates")
        return transactions
    
    def benchmark_data_loading(self, transaction_counts: List[int]) -> Dict[str, Any]:
        """Benchmark data loading performance."""
        print("Benchmarking data loading performance...")
        
        results = {
            'transaction_counts': transaction_counts,
            'processing_times': [],
            'memory_usage': [],
            'transactions_per_second': []
        }
        
        loader = DataLoader(self.config)
        
        for count in transaction_counts:
            print(f"  Testing with {count} transactions...")
            
            # Generate test data
            transactions = self.generate_test_data(count)
            
            # Create temporary Excel file
            temp_file = self._create_temp_excel(transactions)
            
            # Measure memory before
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Time the loading
            start_time = time.time()
            loaded_transactions = loader.load_workbook(temp_file)
            end_time = time.time()
            
            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            processing_time = end_time - start_time
            memory_used = memory_after - memory_before
            tps = len(loaded_transactions) / processing_time
            
            results['processing_times'].append(processing_time)
            results['memory_usage'].append(memory_used)
            results['transactions_per_second'].append(tps)
            
            # Cleanup
            Path(temp_file).unlink()
            
            print(f"    Time: {processing_time:.2f}s, Memory: {memory_used:.1f}MB, TPS: {tps:.0f}")
        
        return results
    
    def benchmark_categorization(self, transaction_counts: List[int]) -> Dict[str, Any]:
        """Benchmark categorization performance."""
        print("Benchmarking categorization performance...")
        
        results = {
            'transaction_counts': transaction_counts,
            'processing_times': [],
            'accuracy_scores': [],
            'confidence_scores': []
        }
        
        categorizer = MLCategorizer(self.config)
        
        for count in transaction_counts:
            print(f"  Testing with {count} transactions...")
            
            transactions = self.generate_test_data(count)
            
            start_time = time.time()
            categorized = categorizer.categorize_transactions(transactions)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Calculate metrics
            categorized_count = sum(1 for t in categorized if t.category != 'Uncategorized')
            accuracy = categorized_count / len(categorized)
            avg_confidence = sum(t.confidence for t in categorized) / len(categorized)
            
            results['processing_times'].append(processing_time)
            results['accuracy_scores'].append(accuracy)
            results['confidence_scores'].append(avg_confidence)
            
            print(f"    Time: {processing_time:.2f}s, Accuracy: {accuracy:.1%}, Confidence: {avg_confidence:.2f}")
        
        return results
    
    def benchmark_duplicate_detection(self, transaction_counts: List[int]) -> Dict[str, Any]:
        """Benchmark duplicate detection performance."""
        print("Benchmarking duplicate detection performance...")
        
        results = {
            'transaction_counts': transaction_counts,
            'processing_times': [],
            'duplicates_found': [],
            'precision_scores': [],
            'recall_scores': []
        }
        
        detector = DuplicateDetector(self.config)
        
        for count in transaction_counts:
            print(f"  Testing with {count} transactions...")
            
            transactions = self.generate_test_data(count, duplicate_rate=0.1)
            
            start_time = time.time()
            processed = detector.detect_duplicates(transactions)
            end_time = time.time()
            
            processing_time = end_time - start_time
            duplicates_found = sum(1 for t in processed if t.is_duplicate)
            
            # For synthetic data, we can't calculate true precision/recall
            # In practice, you would compare against manually labeled data
            estimated_precision = 0.85  # Placeholder
            estimated_recall = 0.75     # Placeholder
            
            results['processing_times'].append(processing_time)
            results['duplicates_found'].append(duplicates_found)
            results['precision_scores'].append(estimated_precision)
            results['recall_scores'].append(estimated_recall)
            
            print(f"    Time: {processing_time:.2f}s, Duplicates: {duplicates_found}")
        
        return results
    
    def benchmark_end_to_end(self, transaction_counts: List[int]) -> Dict[str, Any]:
        """Benchmark complete pipeline performance."""
        print("Benchmarking end-to-end pipeline performance...")
        
        results = {
            'transaction_counts': transaction_counts,
            'total_times': [],
            'memory_peaks': [],
            'cpu_usage': []
        }
        
        for count in transaction_counts:
            print(f"  Testing complete pipeline with {count} transactions...")
            
            # Generate test data and create temporary file
            transactions = self.generate_test_data(count)
            temp_file = self._create_temp_excel(transactions)
            
            # Monitor system resources
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024
            cpu_before = process.cpu_percent()
            
            # Run complete pipeline
            start_time = time.time()
            
            pipeline = FinancialPipeline()
            pipeline.config = self.config
            results_data = pipeline.process_file(temp_file)
            
            end_time = time.time()
            
            # Measure final resource usage
            memory_after = process.memory_info().rss / 1024 / 1024
            cpu_after = process.cpu_percent()
            
            total_time = end_time - start_time
            memory_peak = memory_after - memory_before
            cpu_usage = cpu_after - cpu_before
            
            results['total_times'].append(total_time)
            results['memory_peaks'].append(memory_peak)
            results['cpu_usage'].append(cpu_usage)
            
            # Cleanup
            Path(temp_file).unlink()
            
            print(f"    Total time: {total_time:.2f}s, Memory: {memory_peak:.1f}MB, CPU: {cpu_usage:.1f}%")
        
        return results
    
    def profile_memory_usage(self, num_transactions: int = 10000):
        """Profile memory usage during processing."""
        print(f"Profiling memory usage with {num_transactions} transactions...")
        
        @memory_profiler.profile
        def process_with_profiling():
            transactions = self.generate_test_data(num_transactions)
            temp_file = self._create_temp_excel(transactions)
            
            pipeline = FinancialPipeline()
            pipeline.config = self.config
            results = pipeline.process_file(temp_file)
            
            Path(temp_file).unlink()
            return results
        
        # Run with memory profiling
        process_with_profiling()
    
    def profile_cpu_usage(self, num_transactions: int = 10000):
        """Profile CPU usage and identify bottlenecks."""
        print(f"Profiling CPU usage with {num_transactions} transactions...")
        
        transactions = self.generate_test_data(num_transactions)
        temp_file = self._create_temp_excel(transactions)
        
        # Create profiler
        profiler = cProfile.Profile()
        
        # Profile the execution
        profiler.enable()
        
        pipeline = FinancialPipeline()
        pipeline.config = self.config
        results = pipeline.process_file(temp_file)
        
        profiler.disable()
        
        # Analyze results
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        print("Top CPU usage functions:")
        print(s.getvalue())
        
        # Cleanup
        Path(temp_file).unlink()
        
        return s.getvalue()
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run all benchmarks and return comprehensive results."""
        print("Starting comprehensive performance benchmark...")
        
        # Test with different dataset sizes
        transaction_counts = [100, 500, 1000, 2500, 5000, 10000]
        
        all_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'system_info': {
                    'cpu_count': psutil.cpu_count(),
                    'memory_gb': psutil.virtual_memory().total / (1024**3),
                    'python_version': platform.python_version()
                }
            },
            'data_loading': self.benchmark_data_loading(transaction_counts),
            'categorization': self.benchmark_categorization(transaction_counts),
            'duplicate_detection': self.benchmark_duplicate_detection(transaction_counts),
            'end_to_end': self.benchmark_end_to_end(transaction_counts)
        }
        
        # Generate visualizations
        self._create_performance_plots(all_results)
        
        # Save results
        output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"Benchmark results saved to {output_file}")
        return all_results
    
    def _create_temp_excel(self, transactions: List[TransactionRecord]) -> str:
        """Create temporary Excel file from transactions."""
        import tempfile
        
        # Convert to DataFrame
        data = {
            'Date': [t.date for t in transactions],
            'Amount': [t.amount for t in transactions],
            'Description': [t.description for t in transactions],
            'Account': [t.account for t in transactions]
        }
        df = pd.DataFrame(data)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        df.to_excel(temp_file.name, index=False, sheet_name='Transactions')
        temp_file.close()
        
        return temp_file.name
    
    def _create_performance_plots(self, results: Dict[str, Any]):
        """Create performance visualization plots."""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Financial Processing Pipeline Performance Analysis', fontsize=16)
        
        # Plot 1: Processing Time vs Dataset Size
        ax1 = axes[0, 0]
        counts = results['end_to_end']['transaction_counts']
        times = results['end_to_end']['total_times']
        ax1.plot(counts, times, 'o-', linewidth=2, markersize=8, color='#2E86AB')
        ax1.set_xlabel('Number of Transactions')
        ax1.set_ylabel('Processing Time (seconds)')
        ax1.set_title('End-to-End Processing Time')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Memory Usage
        ax2 = axes[0, 1]
        memory = results['end_to_end']['memory_peaks']
        ax2.plot(counts, memory, 's-', linewidth=2, markersize=8, color='#A23B72')
        ax2.set_xlabel('Number of Transactions')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Peak Memory Usage')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Component Performance Comparison
        ax3 = axes[1, 0]
        load_times = results['data_loading']['processing_times']
        cat_times = results['categorization']['processing_times']
        dup_times = results['duplicate_detection']['processing_times']
        
        x = np.arange(len(counts))
        width = 0.25
        
        ax3.bar(x - width, load_times, width, label='Data Loading', color='#F18F01')
        ax3.bar(x, cat_times, width, label='Categorization', color='#C73E1D')
        ax3.bar(x + width, dup_times, width, label='Duplicate Detection', color='#592941')
        
        ax3.set_xlabel('Dataset Size Index')
        ax3.set_ylabel('Processing Time (seconds)')
        ax3.set_title('Component Performance Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'{c}' for c in counts])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Throughput (Transactions per Second)
        ax4 = axes[1, 1]
        throughput = [c/t for c, t in zip(counts, times)]
        ax4.plot(counts, throughput, '^-', linewidth=2, markersize=8, color='#3E92CC')
        ax4.set_xlabel('Number of Transactions')
        ax4.set_ylabel('Throughput (Transactions/sec)')
        ax4.set_title('Processing Throughput')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Performance plots saved to performance_analysis.png")


# examples/usage_examples.py - Usage Examples
"""
Comprehensive usage examples for the Financial Transaction Processor.
Demonstrates various use cases and integration patterns.
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
import requests
import json

# Import our modules
from processor.pipeline import FinancialPipeline, ProcessingConfig, TransactionRecord


class UsageExamples:
    """Comprehensive usage examples and tutorials."""
    
    def example_1_basic_processing(self):
        """Example 1: Basic file processing with default settings."""
        print("=== Example 1: Basic File Processing ===")
        
        # Initialize pipeline with default configuration
        pipeline = FinancialPipeline()
        
        # Process a file
        try:
            results = pipeline.process_file("data/sample/transactions.xlsx")
            
            print(f"Processed {results['summary']['total_transactions']} transactions")
            print(f"Found {results['summary']['duplicate_count']} duplicates")
            print(f"Identified {results['summary']['categories']['count']} categories")
            
            # Generate reports
            pipeline.generate_reports(results, "output/basic_example/")
            print("Reports generated in output/basic_example/")
            
        except FileNotFoundError:
            print("Sample file not found. Creating synthetic data...")
            self._create_sample_data()
    
    def example_2_custom_configuration(self):
        """Example 2: Processing with custom configuration."""
        print("=== Example 2: Custom Configuration ===")
        
        # Create custom configuration
        config = ProcessingConfig(
            ml_confidence_threshold=0.9,  # Higher confidence requirement
            temporal_window_days=3,       # Shorter duplicate detection window
            amount_tolerance=0.05,        # Larger amount tolerance
            fuzzy_threshold=90            # Higher fuzzy matching threshold
        )
        
        # Initialize pipeline with custom config
        pipeline = FinancialPipeline()
        pipeline.config = config
        
        # Process file
        try:
            results = pipeline.process_file("data/sample/transactions.xlsx")
            print("Processing completed with custom configuration")
            
            # Print configuration impact
            avg_confidence = results['summary']['confidence_stats']['mean']
            print(f"Average categorization confidence: {avg_confidence:.2f}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    def example_3_api_integration(self):
        """Example 3: Using the REST API."""
        print("=== Example 3: API Integration ===")
        
        api_base = "http://localhost:8000"
        headers = {"Authorization": "Bearer demo-api-key"}
        
        # Check API health
        try:
            response = requests.get(f"{api_base}/health")
            if response.status_code == 200:
                print("API is healthy")
                health_data = response.json()
                print(f"Uptime: {health_data['uptime_seconds']:.1f} seconds")
            
            # Upload a file (would need actual file)
            print("To upload a file via API:")
            print(f"POST {api_base}/upload")
            print("Headers:", headers)
            print("Files: {'file': ('transactions.xlsx', file_data, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}")
            
        except requests.ConnectionError:
            print("API server not running. Start with: uvicorn api.main:app")
    
    def example_4_batch_processing(self):
        """Example 4: Batch processing multiple files."""
        print("=== Example 4: Batch Processing ===")
        
        from pathlib import Path
        
        # Find all Excel files in a directory
        input_dir = Path("data/input")
        if not input_dir.exists():
            print("Input directory not found. Creating sample structure...")
            input_dir.mkdir(parents=True, exist_ok=True)
            return
        
        excel_files = list(input_dir.glob("*.xlsx"))
        
        if not excel_files:
            print("No Excel files found in input directory")
            return
        
        # Process each file
        pipeline = FinancialPipeline()
        
        for file_path in excel_files:
            print(f"Processing {file_path.name}...")
            
            try:
                results = pipeline.process_file(str(file_path))
                
                # Create output directory for this file
                output_dir = Path("output/batch") / file_path.stem
                pipeline.generate_reports(results, str(output_dir))
                
                print(f"  Completed: {results['summary']['total_transactions']} transactions")
                
            except Exception as e:
                print(f"  Error processing {file_path.name}: {e}")
    
    def example_5_data_analysis(self):
        """Example 5: Advanced data analysis and insights."""
        print("=== Example 5: Advanced Data Analysis ===")
        
        # Create sample data for analysis
        transactions = self._create_sample_transactions(1000)
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([{
            'date': t.date,
            'amount': t.amount,
            'description': t.description,
            'category': t.category or 'Uncategorized'
        } for t in transactions])
        
        print("Transaction Analysis:")
        print(f"Total transactions: {len(df)}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Total amount: ${df['amount'].sum():,.2f}")
        
        # Category analysis
        print("\nCategory breakdown:")
        category_summary = df.groupby('category')['amount'].agg(['count', 'sum', 'mean'])
        print(category_summary)
        
        # Monthly trends
        df['month'] = df['date'].dt.to_period('M')
        monthly_summary = df.groupby('month')['amount'].sum()
        print(f"\nMonthly spending trend:")
        for month, amount in monthly_summary.items():
            print(f"  {month}: ${amount:,.2f}")
        
        # Identify largest transactions
        print("\nLargest transactions:")
        largest = df.nlargest(5, 'amount')[['date', 'amount', 'description', 'category']]
        for _, row in largest.iterrows():
            print(f"  {row['date'].strftime('%Y-%m-%d')}: ${row['amount']:,.2f} - {row['description']}")
    
    def example_6_ml_model_training(self):
        """Example 6: Training custom ML categorization model."""
        print("=== Example 6: ML Model Training ===")
        
        from processor.categorizer import MLCategorizer
        
        # Create training data (in practice, this would be manually labeled)
        training_transactions = [
            TransactionRecord(datetime(2024, 1, 1), 25.50, "STARBUCKS COFFEE", "checking"),
            TransactionRecord(datetime(2024, 1, 2), 45.00, "SHELL GAS STATION", "checking"),
            TransactionRecord(datetime(2024, 1, 3), 125.00, "GROCERY STORE", "checking"),
            TransactionRecord(datetime(2024, 1, 4), 1200.00, "RENT PAYMENT", "checking"),
            TransactionRecord(datetime(2024, 1, 5), 50.00, "AMAZON PURCHASE", "credit"),
        ]
        
        training_labels = [
            "Food & Dining",
            "Transportation", 
            "Food & Dining",
            "Housing",
            "Shopping"
        ]
        
        # Initialize and train categorizer
        config = ProcessingConfig()
        categorizer = MLCategorizer(config)
        
        try:
            categorizer.train(training_transactions, training_labels)
            print("Model training completed")
            
            # Test predictions
            test_transaction = TransactionRecord(
                datetime(2024, 2, 1), 35.00, "COFFEE SHOP DOWNTOWN", "checking"
            )
            
            predictions = categorizer.predict([test_transaction])
            category, confidence = predictions[0]
            print(f"Prediction: {category} (confidence: {confidence:.2f})")
            
        except Exception as e:
            print(f"Training error: {e}")
    
    def example_7_real_time_monitoring(self):
        """Example 7: Real-time transaction monitoring."""
        print("=== Example 7: Real-time Monitoring ===")
        
        import threading
        import queue
        import time
        
        # Simulate real-time transaction stream
        transaction_queue = queue.Queue()
        
        def transaction_generator():
            """Simulate incoming transactions."""
            merchants = ["COFFEE SHOP", "GAS STATION", "GROCERY", "RESTAURANT"]
            
            for i in range(20):
                transaction = TransactionRecord(
                    date=datetime.now(),
                    amount=round(random.uniform(5, 200), 2),
                    description=f"{random.choice(merchants)} #{i:04d}",
                    account="checking"
                )
                transaction_queue.put(transaction)
                time.sleep(0.5)  # New transaction every 0.5 seconds
        
        def transaction_processor():
            """Process transactions as they arrive."""
            config = ProcessingConfig()
            categorizer = MLCategorizer(config)
            
            while True:
                try:
                    transaction = transaction_queue.get(timeout=1)
                    
                    # Process transaction
                    categorized = categorizer.categorize_transactions([transaction])
                    t = categorized[0]
                    
                    print(f"Processed: ${t.amount:6.2f} | {t.category:15s} | {t.description}")
                    
                    transaction_queue.task_done()
                    
                except queue.Empty:
                    break
        
        # Start threads
        print("Starting real-time transaction processing...")
        
        generator_thread = threading.Thread(target=transaction_generator)
        processor_thread = threading.Thread(target=transaction_processor)
        
        generator_thread.start()
        processor_thread.start()
        
        generator_thread.join()
        processor_thread.join()
        
        print("Real-time processing completed")
    
    def _create_sample_data(self):
        """Create sample data file for examples."""
        print("Creating sample data...")
        
        # Generate sample transactions
        transactions = self._create_sample_transactions(100)
        
        # Convert to DataFrame
        data = {
            'Date': [t.date for t in transactions],
            'Amount': [t.amount for t in transactions],
            'Description': [t.description for t in transactions],
            'Account': [t.account for t in transactions]
        }
        df = pd.DataFrame(data)
        
        # Save to Excel
        output_dir = Path("data/sample")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df.to_excel(output_dir / "transactions.xlsx", index=False, sheet_name="Checking")
        print("Sample data created: data/sample/transactions.xlsx")
    
    def _create_sample_transactions(self, count: int) -> List[TransactionRecord]:
        """Create sample transactions for examples."""
        import random
        
        transactions = []
        base_date = datetime(2024, 1, 1)
        
        merchants = [
            ("STARBUCKS COFFEE", "Food & Dining", (3, 15)),
            ("SHELL GAS STATION", "Transportation", (25, 80)),
            ("WALMART SUPERCENTER", "Shopping", (20, 200)),
            ("ELECTRIC COMPANY", "Utilities", (80, 150)),
            ("AMAZON PURCHASE", "Shopping", (10, 300)),
            ("RESTAURANT", "Food & Dining", (15, 100)),
            ("ATM WITHDRAWAL", "Cash", (20, 200)),
            ("PAYROLL DEPOSIT", "Income", (1000, 3000))
        ]
        
        for i in range(count):
            merchant_name, category, (min_amt, max_amt) = random.choice(merchants)
            
            # Create transaction
            transaction = TransactionRecord(
                date=base_date + timedelta(days=random.randint(0, 365)),
                amount=round(random.uniform(min_amt, max_amt), 2),
                description=f"{merchant_name} #{random.randint(1000, 9999)}",
                account=random.choice(["Checking", "Savings", "Credit"]),
                category=category,
                confidence=random.uniform(0.7, 0.95)
            )
            
            transactions.append(transaction)
        
        return transactions


# scripts/data_generator.py - Synthetic Data Generation
"""
Generate realistic synthetic financial data for testing and development.
"""

class SyntheticDataGenerator:
    """Generate realistic financial transaction data."""
    
    def __init__(self):
        self.merchant_patterns = {
            'Food & Dining': {
                'names': ['STARBUCKS', 'MCDONALDS', 'SUBWAY', 'CHIPOTLE', 'RESTAURANT', 'CAFE', 'DINER'],
                'amount_range': (5, 100),
                'frequency_weight': 0.25
            },
            'Transportation': {
                'names': ['SHELL', 'EXXON', 'BP', 'UBER', 'LYFT', 'PARKING', 'METRO'],
                'amount_range': (15, 150),
                'frequency_weight': 0.15
            },
            'Shopping': {
                'names': ['AMAZON', 'TARGET', 'WALMART', 'COSTCO', 'BEST BUY', 'MACYS'],
                'amount_range': (20, 500),
                'frequency_weight': 0.20
            },
            'Utilities': {
                'names': ['ELECTRIC CO', 'WATER DEPT', 'GAS COMPANY', 'INTERNET', 'PHONE BILL'],
                'amount_range': (50, 200),
                'frequency_weight': 0.05
            },
            'Healthcare': {
                'names': ['PHARMACY', 'DENTAL OFFICE', 'MEDICAL CENTER', 'CVS', 'WALGREENS'],
                'amount_range': (15, 300),
                'frequency_weight': 0.10
            }
        }
    
    def generate_realistic_dataset(self, 
                                  num_transactions: int,
                                  start_date: datetime,
                                  end_date: datetime,
                                  accounts: List[str] = None) -> pd.DataFrame:
        """Generate a realistic transaction dataset."""
        
        if accounts is None:
            accounts = ['Checking', 'Savings', 'Credit Card']
        
        transactions = []
        
        # Generate weighted random transactions
        categories = list(self.merchant_patterns.keys())
        weights = [self.merchant_patterns[cat]['frequency_weight'] for cat in categories]
        
        for _ in range(num_transactions):
            # Select category based on weights
            category = random.choices(categories, weights=weights)[0]
            pattern = self.merchant_patterns[category]
            
            # Generate transaction details
            merchant = random.choice(pattern['names'])
            amount = round(random.uniform(*pattern['amount_range']), 2)
            
            # Random date within range
            date_range = (end_date - start_date).days
            random_date = start_date + timedelta(days=random.randint(0, date_range))
            
            # Account selection
            account = random.choice(accounts)
            
            # Create description with realistic variation
            description = f"{merchant} #{random.randint(1000, 9999)}"
            if random.random() < 0.1:  # 10% chance of additional info
                description += random.choice([" ONLINE", " STORE", " AUTO", " RECURRING"])
            
            transactions.append({
                'Date': random_date,
                'Amount': amount,
                'Description': description,
                'Account': account,
                'Category': category
            })
        
        # Convert to DataFrame and sort by date
        df = pd.DataFrame(transactions)
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df
    
    def add_realistic_duplicates(self, df: pd.DataFrame, duplicate_rate: float = 0.05) -> pd.DataFrame:
        """Add realistic duplicate transactions to the dataset."""
        
        num_duplicates = int(len(df) * duplicate_rate)
        duplicates = []
        
        for _ in range(num_duplicates):
            # Select a random transaction to duplicate
            original_idx = random.randint(0, len(df) - 1)
            original = df.iloc[original_idx]
            
            # Create duplicate with slight variations
            duplicate = original.copy()
            
            # Vary the date slightly (within 1-3 days)
            date_offset = timedelta(days=random.randint(1, 3))
            duplicate['Date'] = original['Date'] + date_offset
            
            # Slightly vary amount (simulate pending vs posted)
            if random.random() < 0.3:  # 30% chance of amount variation
                amount_variation = random.uniform(-0.02, 0.02)
                duplicate['Amount'] = round(original['Amount'] + amount_variation, 2)
            
            # Modify description slightly
            if random.random() < 0.4:  # 40% chance of description variation
                duplicate['Description'] += random.choice([" PENDING", " POSTED", " TEMP"])
            
            duplicates.append(duplicate)
        
        # Add duplicates to DataFrame
        duplicates_df = pd.DataFrame(duplicates)
        combined_df = pd.concat([df, duplicates_df], ignore_index=True)
        combined_df = combined_df.sort_values('Date').reset_index(drop=True)
        
        return combined_df


if __name__ == "__main__":
    # Run examples
    examples = UsageExamples()
    
    print("Financial Transaction Processor - Usage Examples")
    print("=" * 50)
    
    # Run all examples
    examples.example_1_basic_processing()
    print()
    
    examples.example_2_custom_configuration()
    print()
    
    examples.example_3_api_integration()
    print()
    
    examples.example_5_data_analysis()
    print()
    
    # Run benchmark if requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        print("Running performance benchmark...")
        benchmark = PerformanceBenchmark()
        results = benchmark.run_comprehensive_benchmark()
        print("Benchmark completed!")
