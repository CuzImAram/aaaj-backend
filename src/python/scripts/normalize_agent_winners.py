import json
import os
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def normalize_winner(value):
    if not isinstance(value, str):
        return value, False

    original_value = value
    normalized_value = value.strip().lower()

    if normalized_value == 'a':
        return 'a', False # Already correct (ignoring case fix if it was 'A', but user said caps ignored)
    if normalized_value == 'b':
        return 'b', False
    if normalized_value == 'n':
        return 'n', False

    # Replacements
    if 'response a' in normalized_value:
         # Check if it is exactly "response a" or similar enough?
         # User said: "Response a" -> a.
         # Let's handle exact matches or mostly exact matches.
         # "Response a " might happen.
         if normalized_value == 'response a':
             return 'a', True

    if 'response b' in normalized_value:
        if normalized_value == 'response b':
            return 'b', True

    if normalized_value == 'tie':
        return 'n', True

    return original_value, False

def process_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read {file_path}: {e}")
        return

    changed = False

    # Data is expected to be a list of dicts, or a dict.
    # Based on context: [ { ... } ]

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = [data]
    else:
        logging.warning(f"Unexpected JSON structure in {os.path.basename(file_path)}")
        return

    for item in items:
        # iterate over all keys in the item that have "agent_winner"
        for key, value in item.items():
            if isinstance(value, dict) and "agent_winner" in value:
                current_winner = value["agent_winner"]

                # Normalize logic
                normalized_val = current_winner.lower().strip()
                new_winner = current_winner

                if normalized_val in ['a', 'b', 'n']:
                    # Ensure it is lowercase a, b, n
                    if current_winner != normalized_val:
                         new_winner = normalized_val
                         logging.info(f"File {os.path.basename(file_path)}: Key '{key}' - Normalizing case '{current_winner}' -> '{new_winner}'")
                         changed = True
                else:
                    # Apply specific rules
                    if normalized_val == 'response a':
                        new_winner = 'a'
                        logging.info(f"File {os.path.basename(file_path)}: Key '{key}' - Replacing '{current_winner}' -> '{new_winner}'")
                        changed = True
                    elif normalized_val == 'response b':
                        new_winner = 'b'
                        logging.info(f"File {os.path.basename(file_path)}: Key '{key}' - Replacing '{current_winner}' -> '{new_winner}'")
                        changed = True
                    elif normalized_val == 'tie':
                        new_winner = 'n'
                        logging.info(f"File {os.path.basename(file_path)}: Key '{key}' - Replacing '{current_winner}' -> '{new_winner}'")
                        changed = True
                    else:
                        logging.info(f"File {os.path.basename(file_path)}: Key '{key}' - Unknown value found: '{current_winner}'")

                if new_winner != current_winner:
                     value["agent_winner"] = new_winner
                     changed = True

    if changed:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2) # Use indent=2 to match existing style if possible, or 4. Original file had indent 2 in provided context.
            # logging.info(f"Saved changes to {os.path.basename(file_path)}")
        except Exception as e:
            logging.error(f"Failed to write to {file_path}: {e}")

def main():
    # Construct path to the directory
    # Script is in src/python/scripts/
    # Target is data/output/compared_ratings_agent_comp/

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    target_dir = os.path.join(base_dir, 'data', 'output', 'compared_ratings_agent_comp')

    if not os.path.exists(target_dir):
        logging.error(f"Directory not found: {target_dir}")
        return

    logging.info(f"Scanning directory: {target_dir}")

    json_files = glob.glob(os.path.join(target_dir, "*.json"))
    logging.info(f"Found {len(json_files)} JSON files.")

    for json_file in json_files:
        process_file(json_file)

    logging.info("Done.")

if __name__ == "__main__":
    main()

