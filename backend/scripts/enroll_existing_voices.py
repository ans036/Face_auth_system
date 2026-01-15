
import sys
import os
import glob
import numpy as np

# Ensure /app is in path to import core modules
sys.path.append("/app")

from core.voice_embedder import get_voice_embedder
from db.crud import create_voice_entry, get_voice_stats, clear_voice_gallery

def main():
    print("🔄 Initializing Voice Enrollment Script...")
    
    # Optional: clear existing voice entries to avoid duplicates?
    # For now, let's keep them and maybe clear if requested. 
    # But to be safe against duplicates, maybe we should check if exists?
    # Our DB model likely has ID primary key, so we'd just add more rows.
    # Let's clear for a clean state since this is a fix.
    
    print("🧹 Clearing existing voice gallery to ensure clean state...")
    clear_voice_gallery()
    
    print("🔍 Scanning /app/database for voice samples...")
    try:
        voice_embedder = get_voice_embedder()
    except Exception as e:
        print(f"❌ Failed to initialize VoiceEmbedder: {e}")
        return

    # Use os.walk or glob
    base_dir = "/app/database"
    
    count = 0
    success = 0
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_path = os.path.join(root, file)
                
                # Extract username from parent folder name
                # /app/database/Username/voice1.wav
                parent_dir = os.path.basename(root)
                username = parent_dir
                
                # specific check to ignore if parent is 'database' or handled folders
                if username == "database" or username == "":
                    continue

                print(f"🎤 Processing {file} for user '{username}'...")
                
                try:
                    with open(wav_path, "rb") as f:
                        content = f.read()
                    
                    if len(content) < 1000:
                        print(f"⚠️ Skipping {file}: Too small ({len(content)} bytes)")
                        continue

                    emb = voice_embedder.embed_from_bytes(content)
                    
                    if emb is not None:
                        create_voice_entry(username, emb)
                        print(f"✅ Enrolled {file} for {username}")
                        success += 1
                    else:
                        print(f"⚠️ Embedder returned None for {file} (Quality issue?)")
                        
                except Exception as e:
                    print(f"❌ Error enrolling {file}: {e}")
                
                count += 1
    
    print("-" * 30)
    print(f"🎉 Enrollment Complete!")
    print(f"Total Files Scanned: {count}")
    print(f"Successful Enrollments: {success}")
    
    stats = get_voice_stats()
    print("📊 Current Voice Stats (Samples per User):", stats)

if __name__ == "__main__":
    main()
