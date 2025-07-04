# src/utils/preprocess.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

class TimeSeriesPreprocessor:
    def __init__(self, sequence_length: int = 10, test_size: float = 0.2):
        """
        Initialize preprocessor with configuration
        
        Args:
            sequence_length: Number of time steps to use for each sample
            test_size: Fraction of data to use for testing (0.1-0.3)
        """
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = sequence_length
        self.test_size = test_size
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load and validate CSV data"""
        df = pd.read_csv(filepath)
        
        # Validate required columns
        if 'Date' not in df.columns or 'Value' not in df.columns:
            raise ValueError("CSV must contain 'Date' and 'Value' columns")
            
        # Convert date and sort
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df

    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time-series sequences for LSTM
        
        Args:
            data: Scaled 1D numpy array of values
            
        Returns:
            X: Sequences of shape (n_samples, sequence_length, 1)
            y: Targets of shape (n_samples,)
        """
        X, y = [], []
        for i in range(len(data)-self.sequence_length-1):
            X.append(data[i:(i+self.sequence_length), 0])
            y.append(data[i+self.sequence_length, 0])
            
        return np.array(X), np.array(y)

    def preprocess(self, filepath: str) -> dict:
        """
        Complete preprocessing pipeline
        
        Returns:
            Dictionary containing:
            - X_train, y_train: Training data
            - X_test, y_test: Testing data
            - scaler: Fitted scaler for inverse transforms
            - df: Original dataframe
        """
        # Load and validate data
        df = self.load_data(filepath)
        values = df['Value'].values.reshape(-1, 1)
        
        # Scale data
        scaled_data = self.scaler.fit_transform(values)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        
        # Split into train/test (maintaining temporal order)
        test_samples = int(len(X) * self.test_size)
        X_train, X_test = X[:-test_samples], X[-test_samples:]
        y_train, y_test = y[:-test_samples], y[-test_samples:]
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'scaler': self.scaler,
            'df': df
        }

# Example usage
if __name__ == "__main__":
    preprocessor = TimeSeriesPreprocessor(sequence_length=10)
    processed = preprocessor.preprocess("../data/data.csv")
    
    print(f"Training shapes: X={processed['X_train'].shape}, y={processed['y_train'].shape}")
    print(f"Test shapes: X={processed['X_test'].shape}, y={processed['y_test'].shape}")