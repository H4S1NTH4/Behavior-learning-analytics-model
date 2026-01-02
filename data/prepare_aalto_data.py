import zipfile
import numpy as np
import os
import time
import csv
import io
import sys

# FIX: Increase the CSV field size limit to handle massive lines
csv.field_size_limit(sys.maxsize)

# --- CONFIGURATION ---
ZIP_FILENAME = 'Keystrokes.zip' 
OUTPUT_FILENAME = 'processed_aalto_data.npy'
TARGET_USER_COUNT = 15000
MIN_SEQUENCE_LENGTH = 70 # Standard TypeNet length

def extract_features_from_txt(tsv_content):
    """
    Parses Aalto TSV content and converts it to TypeNet features.
    Returns: List of Numpy arrays (one per sentence session)
    """
    # Use python's CSV module to handle tab-separated lines robustly
    reader = csv.reader(tsv_content, delimiter='\t')
    
    # Skip header if it exists (usually "PARTICIPANT_ID...")
    try:
        header = next(reader) 
    except StopIteration:
        return []

    # We need to group keystrokes by their sentence (TEST_SECTION_ID)
    # Structure: { section_id: [ list of raw events ] }
    sessions = {}
    
    for row in reader:
        # Safety check for malformed rows
        if len(row) < 9: continue 
        
        try:
            section_id = row[1]
            press_time = int(row[5])
            release_time = int(row[6])
            keycode = int(row[8])
            
            if section_id not in sessions:
                sessions[section_id] = []
            
            sessions[section_id].append({
                'p': press_time, 
                'r': release_time, 
                'k': keycode
            })
        except ValueError:
            continue # Skip rows with bad numbers

    # Now convert each session into TypeNet features
    all_session_features = []
    
    for section_id, events in sessions.items():
        # Sort by press time to be safe
        events.sort(key=lambda x: x['p'])
        
        if len(events) < MIN_SEQUENCE_LENGTH: continue

        features = []
        for i in range(len(events) - 1):
            curr = events[i]
            next_k = events[i+1]
            
            # CALCULATE THE 5 FEATURES (ms -> seconds)
            # 1. Hold Time (Release - Press)
            hl = (curr['r'] - curr['p']) / 1000.0
            
            # 2. Flight Time (Next Press - Current Release)
            il = (next_k['p'] - curr['r']) / 1000.0
            
            # 3. Press-to-Press (Next Press - Current Press)
            pl = (next_k['p'] - curr['p']) / 1000.0
            
            # 4. Release-to-Release (Next Release - Current Release)
            rl = (next_k['r'] - curr['r']) / 1000.0
            
            # 5. KeyCode (Normalized)
            code = curr['k'] / 255.0
            
            features.append([hl, il, pl, rl, code])
            
            if len(features) == MIN_SEQUENCE_LENGTH:
                break
        
        # Only keep full-length sequences
        if len(features) == MIN_SEQUENCE_LENGTH:
            all_session_features.append(np.array(features, dtype=np.float32))

    return all_session_features

def main():
    if not os.path.exists(ZIP_FILENAME):
        print(f"\u274c Error: Could not find '{ZIP_FILENAME}'")
        return

    print(f"\U0001f4c2 Reading {ZIP_FILENAME}...")
    start_time = time.time()
    
    collected_data = [] # List of users, each user is a list of sequences
    users_processed = 0
    
    with zipfile.ZipFile(ZIP_FILENAME, 'r') as z:
        # Filter specifically for the keystroke text files
        # The README says format is "<number>_keystrokes.txt"
        file_list = [f for f in z.namelist() if "_keystrokes.txt" in f and "__MACOSX" not in f]
        
        print(f"\U0001f50d Found {len(file_list)} user log files.")
        print(f"\U0001f680 Processing first {TARGET_USER_COUNT} valid users...")

        for file_path in file_list:
            if users_processed >= TARGET_USER_COUNT:
                break
            
            with z.open(file_path) as f:
                # Read bytes, decode to string for CSV reader
                text_content = io.TextIOWrapper(f, encoding='utf-8', errors='replace')
                
                # Extract all valid sequences for this user
                user_sequences = extract_features_from_txt(text_content)
                
                # We need users with at least 5 valid sessions for the 'Few-Shot' training
                if len(user_sequences) >= 5:
                    # Take exactly 5 sequences to keep the tensor shape uniform
                    collected_data.append(user_sequences[:5])
                    users_processed += 1
                    
                    if users_processed % 100 == 0:
                        print(f"   \u2705 Processed {users_processed} users...")

    # Convert to one massive Numpy array
    # Shape: (Num_Users, 5_Sessions, 70_Keystrokes, 5_Features)
    final_dataset = np.array(collected_data, dtype=np.float32)
    
    print(f"\n\U0001f389 Finished!")
    print(f"   Collected Users: {len(final_dataset)}")
    print(f"   Final Tensor Shape: {final_dataset.shape}")
    print(f"   Time Taken: {int(time.time() - start_time)} seconds")
    
    print(f"\U0001f4be Saving to {OUTPUT_FILENAME}...")
    np.save(OUTPUT_FILENAME, final_dataset)
    print("\u2705 Done. You can now upload this .npy file to Google Drive.")

if __name__ == "__main__":
    main()