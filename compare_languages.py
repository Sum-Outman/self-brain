import json
import os

def load_json_file(file_path):
    """Load a JSON file and return its content as a dictionary."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

def compare_json_keys(file1_path, file2_path):
    """Compare keys between two JSON files and return missing keys in each."""
    data1 = load_json_file(file1_path)
    data2 = load_json_file(file2_path)
    
    keys1 = set(data1.keys())
    keys2 = set(data2.keys())
    
    missing_in_file2 = keys1 - keys2
    missing_in_file1 = keys2 - keys1
    
    return missing_in_file1, missing_in_file2

def main():
    # Paths to language files
    en_path = "language_resources/en.json"
    zh_path = "language_resources/zh.json"
    
    print("Comparing language resource files...")
    print("=" * 50)
    
    # Compare en.json and zh.json
    missing_in_zh, missing_in_en = compare_json_keys(en_path, zh_path)
    
    print(f"Keys missing in zh.json (compared to en.json): {len(missing_in_zh)}")
    if missing_in_zh:
        print("Missing keys:")
        for key in sorted(missing_in_zh):
            print(f"  - {key}")
    
    print(f"\nKeys missing in en.json (compared to zh.json): {len(missing_in_en)}")
    if missing_in_en:
        print("Missing keys:")
        for key in sorted(missing_in_en):
            print(f"  - {key}")
    
    # Check other language files for completeness
    print("\n" + "=" * 50)
    print("Checking other language files for key count:")
    
    languages = {
        'de': de_path,
        'ja': ja_path,
        'ru': ru_path
    }
    
    en_data = load_json_file(en_path)
    zh_data = load_json_file(zh_path)
    en_key_count = len(en_data.keys())
    zh_key_count = len(zh_data.keys())
    
    print(f"en.json keys: {en_key_count}")
    print(f"zh.json keys: {zh_key_count}")
    
    for lang, path in languages.items():
        data = load_json_file(path)
        key_count = len(data.keys())
        print(f"{lang}.json keys: {key_count} (completeness: {key_count/zh_key_count*100:.1f}%)")
        
        # Show a sample of keys if the file is incomplete
        if key_count < zh_key_count:
            sample_keys = list(data.keys())[:5] if key_count > 5 else list(data.keys())
            print(f"  Sample keys: {sample_keys}")

if __name__ == "__main__":
    main()
