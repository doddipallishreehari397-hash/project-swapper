import os
import time
import logging

CACHE_DIR = "temp_cache"
MAX_AGE_SECONDS = 120 * 60  # 2 Hours
CHECK_INTERVAL_SECONDS = 600  # 10 Minutes

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def clean_expired_cache():
    if not os.path.exists(CACHE_DIR):
        return
    
    current_time = time.time()
    logging.info("Running automated cache cleanup sweep...")
    
    for filename in os.listdir(CACHE_DIR):
        file_path = os.path.join(CACHE_DIR, filename)
        if os.path.isfile(file_path):
            file_creation_time = os.path.getmtime(file_path)
            age = current_time - file_creation_time
            
            if age > MAX_AGE_SECONDS:
                try:
                    os.remove(file_path)
                    logging.info(f"Purged expired cache item: {filename} (Age: {round(age/60, 1)} mins)")
                except Exception as e:
                    logging.error(f"Failed to delete {filename}: {e}")

if __name__ == "__main__":
    logging.info("Cache cleaner background daemon activated.")
    while True:
        clean_expired_cache()
        time.sleep(CHECK_INTERVAL_SECONDS)