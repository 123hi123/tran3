#!/usr/bin/env python3
import pandas as pd
import numpy as np
import random
import argparse

def generate_fake_data(num_ids=5, frames_per_id=50, missing_rate=0.3):
    """
    Generate fake sign language data with mediapipe coordinates
    
    Args:
        num_ids: Number of different video IDs
        frames_per_id: Number of frames per video ID
        missing_rate: Rate of missing values (NaN values)
    """
    
    columns = [
        'id', 'sign_language',
        'nose_x', 'nose_y', 'nose_z',
        'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z',
        'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z'
    ]
    
    for hand in ['left_hand', 'right_hand']:
        for point in range(21):
            for coord in ['x', 'y', 'z']:
                columns.append(f'{hand}_{point}_{coord}')
    
    sign_languages = ['HELLO', 'THANK_YOU', 'PLEASE', 'YES', 'NO', 'SORRY', 'GOOD', 'BAD']
    
    data = []
    
    for video_id in range(1, num_ids + 1):
        sign_lang = random.choice(sign_languages)
        
        for frame in range(frames_per_id):
            row = {
                'id': video_id,
                'sign_language': sign_lang
            }
            
            for col in columns[2:]:
                if random.random() < missing_rate:
                    row[col] = np.nan
                else:
                    if 'x' in col:
                        row[col] = round(random.uniform(0.0, 1.0), 6)
                    elif 'y' in col:
                        row[col] = round(random.uniform(0.0, 1.0), 6)
                    else:  # z coordinate
                        row[col] = round(random.uniform(-0.1, 0.1), 6)
            
            data.append(row)
    
    df = pd.DataFrame(data)
    return df

def main():
    parser = argparse.ArgumentParser(description='Generate fake sign language data')
    parser.add_argument('--ids', type=int, default=5, help='Number of video IDs')
    parser.add_argument('--frames', type=int, default=50, help='Frames per ID')
    parser.add_argument('--missing', type=float, default=0.3, help='Missing value rate')
    parser.add_argument('--output', type=str, default='test_data.csv', help='Output filename')
    
    args = parser.parse_args()
    
    print(f"Generating fake data with {args.ids} IDs, {args.frames} frames each...")
    df = generate_fake_data(args.ids, args.frames, args.missing)
    
    df.to_csv(args.output, index=False)
    print(f"Generated data saved to {args.output}")
    print(f"Shape: {df.shape}")
    print(f"Sample data:")
    print(df.head())

if __name__ == "__main__":
    main()