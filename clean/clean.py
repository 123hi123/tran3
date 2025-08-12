#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import os
from typing import List, Dict, Tuple

def load_sord_mapping(sord_path: str) -> Dict[str, str]:
    """Load the sign language mapping from sord.csv"""
    df = pd.read_csv(sord_path, index_col=0)
    mapping = {}
    for col in df.columns:
        value = df[col].iloc[0] if len(df) > 0 else 's'
        mapping[col] = value
    return mapping

def get_hand_columns(df: pd.DataFrame, hand: str) -> List[str]:
    """Get all columns for a specific hand"""
    hand_cols = []
    for point in range(21):
        for coord in ['x', 'y', 'z']:
            col = f'{hand}_hand_{point}_{coord}'
            if col in df.columns:
                hand_cols.append(col)
    return hand_cols

def calculate_missing_rate(df: pd.DataFrame, columns: List[str]) -> float:
    """Calculate missing rate for given columns (NaN values count as missing)"""
    if not columns:
        return 0.0
    total_values = len(df) * len(columns)
    missing_values = df[columns].isna().sum().sum()
    return missing_values / total_values

def remove_high_missing_hand_data(df: pd.DataFrame, sign_mapping: Dict[str, str], threshold: float = 0.8) -> pd.DataFrame:
    """Remove hand data if missing rate >= threshold for 's' type signs"""
    df_clean = df.copy()
    
    for sign in df['sign_language'].unique():
        if sign in sign_mapping and sign_mapping[sign] == 's':
            sign_mask = df_clean['sign_language'] == sign
            sign_data = df_clean[sign_mask]
            
            left_hand_cols = get_hand_columns(df_clean, 'left')
            right_hand_cols = get_hand_columns(df_clean, 'right')
            
            left_missing_rate = calculate_missing_rate(sign_data, left_hand_cols)
            right_missing_rate = calculate_missing_rate(sign_data, right_hand_cols)
            
            print(f"Sign: {sign} - Left hand missing: {left_missing_rate:.2%}, Right hand missing: {right_missing_rate:.2%}")
            
            if left_missing_rate >= threshold:
                print(f"Removing left hand data for sign: {sign}")
                df_clean.loc[sign_mask, left_hand_cols] = np.nan
            
            if right_missing_rate >= threshold:
                print(f"Removing right hand data for sign: {sign}")
                df_clean.loc[sign_mask, right_hand_cols] = np.nan
    
    return df_clean

def interpolate_missing_values(series: pd.Series) -> pd.Series:
    """Interpolate missing values (NaN) in a series using forward/backward fill and linear interpolation"""
    series_clean = series.copy()
    
    nan_mask = series_clean.isna()
    if not nan_mask.any():
        return series_clean
    
    non_nan_indices = np.where(~nan_mask)[0]
    if len(non_nan_indices) == 0:
        return series_clean
    
    for i in range(len(series_clean)):
        if nan_mask.iloc[i]:
            if i == 0:
                next_valid_idx = non_nan_indices[non_nan_indices > i]
                if len(next_valid_idx) > 0:
                    series_clean.iloc[i] = series_clean.iloc[next_valid_idx[0]]
            elif i == len(series_clean) - 1:
                prev_valid_idx = non_nan_indices[non_nan_indices < i]
                if len(prev_valid_idx) > 0:
                    series_clean.iloc[i] = series_clean.iloc[prev_valid_idx[-1]]
            else:
                prev_valid_idx = non_nan_indices[non_nan_indices < i]
                next_valid_idx = non_nan_indices[non_nan_indices > i]
                
                if len(prev_valid_idx) > 0 and len(next_valid_idx) > 0:
                    prev_val = series_clean.iloc[prev_valid_idx[-1]]
                    next_val = series_clean.iloc[next_valid_idx[0]]
                    series_clean.iloc[i] = (prev_val + next_val) / 2
                elif len(prev_valid_idx) > 0:
                    series_clean.iloc[i] = series_clean.iloc[prev_valid_idx[-1]]
                elif len(next_valid_idx) > 0:
                    series_clean.iloc[i] = series_clean.iloc[next_valid_idx[0]]
    
    return series_clean

def process_nose_shoulder_interpolation(df: pd.DataFrame) -> pd.DataFrame:
    """Process nose and shoulder missing value interpolation for each video ID"""
    df_clean = df.copy()
    
    nose_shoulder_cols = ['nose_x', 'nose_y', 'nose_z', 
                         'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z',
                         'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z']
    
    nose_shoulder_cols = [col for col in nose_shoulder_cols if col in df.columns]
    
    for video_id in df['id'].unique():
        id_mask = df_clean['id'] == video_id
        for col in nose_shoulder_cols:
            df_clean.loc[id_mask, col] = interpolate_missing_values(df_clean.loc[id_mask, col])
    
    return df_clean

def process_group(group_df: pd.DataFrame, sign_mapping: Dict[str, str]) -> pd.DataFrame:
    """Process a single group (video ID) of data"""
    print(f"Processing ID: {group_df['id'].iloc[0]}, Frames: {len(group_df)}")
    
    group_clean = remove_high_missing_hand_data(group_df, sign_mapping)
    
    group_clean = process_nose_shoulder_interpolation(group_clean)
    
    group_clean = group_clean.fillna(0)
    
    return group_clean

def process_data(input_file: str, sord_file: str, low_memory: bool = False) -> pd.DataFrame:
    """Main processing function"""
    print(f"Loading data from {input_file}")
    
    sign_mapping = load_sord_mapping(sord_file)
    print(f"Loaded {len(sign_mapping)} sign language mappings")
    
    if low_memory:
        print("Processing in low memory mode (one group at a time)")
        all_processed_data = []
        
        chunk_reader = pd.read_csv(input_file, chunksize=1000)
        current_group = pd.DataFrame()
        current_id = None
        
        for chunk in chunk_reader:
            for video_id in chunk['id'].unique():
                id_data = chunk[chunk['id'] == video_id]
                
                if current_id is None:
                    current_id = video_id
                    current_group = id_data.copy()
                elif current_id == video_id:
                    current_group = pd.concat([current_group, id_data], ignore_index=True)
                else:
                    processed_group = process_group(current_group, sign_mapping)
                    all_processed_data.append(processed_group)
                    
                    current_id = video_id
                    current_group = id_data.copy()
        
        if len(current_group) > 0:
            processed_group = process_group(current_group, sign_mapping)
            all_processed_data.append(processed_group)
        
        result = pd.concat(all_processed_data, ignore_index=True)
    
    else:
        print("Processing all data at once")
        df = pd.read_csv(input_file)
        result = df.groupby('id').apply(lambda x: process_group(x, sign_mapping)).reset_index(drop=True)
    
    return result

def main():
    # 預設輸入檔案路徑 - 可以修改這裡來設定常用的檔案路徑
    default_input_file = "test_data.csv"  # 修改為你的預設檔案路徑
    
    parser = argparse.ArgumentParser(description='Clean sign language pose data')
    parser.add_argument('input_file', nargs='?', default=default_input_file, help=f'Input CSV file path (default: {default_input_file})')
    parser.add_argument('--sord', default='sord.csv', help='Path to sord.csv mapping file')
    parser.add_argument('--low', action='store_true', help='Low memory mode - process one ID group at a time')
    parser.add_argument('--output', help='Output file path (default: output/cleaned_<input_filename>)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found")
        return
    
    if not os.path.exists(args.sord):
        print(f"Error: Sord file {args.sord} not found")
        return
    
    if args.output is None:
        input_basename = os.path.basename(args.input_file)
        name_without_ext = os.path.splitext(input_basename)[0]
        args.output = f"output/cleaned_{name_without_ext}.csv"
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print("Starting data cleaning process...")
    cleaned_data = process_data(args.input_file, args.sord, args.low)
    
    print(f"Saving cleaned data to {args.output}")
    cleaned_data.to_csv(args.output, index=False)
    
    print("Data cleaning completed!")
    print(f"Original shape: {pd.read_csv(args.input_file).shape}")
    print(f"Cleaned shape: {cleaned_data.shape}")

if __name__ == "__main__":
    main()