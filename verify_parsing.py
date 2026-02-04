import json
import os
import re
import ast

# Exact copy from Scraping.py (as seen in view_file)
def robust_json_parse(text):
    text = text.strip()
    text = text.replace("“", '"').replace("”", '"')
    
    try:
        return json.loads(text)
    except:
        pass
    try:
        # Regex for JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_str = match.group()
            return json.loads(json_str)
    except:
        pass
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        clean_text = match.group() if match else text
        return ast.literal_eval(clean_text)
    except:
        pass
    return None

def test_parsing():
    debug_file = 'debug_gemini_responses.json'
    if not os.path.exists(debug_file):
        print("Debug file not found.")
        return

    with open(debug_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries.")
    
    success_count = 0
    fail_count = 0
    
    for i, entry in enumerate(data[:20]):
        raw = entry.get('raw_response', '')
        # DO NOT CLEAN MANUALLY. Test robust_json_parse capability.
        
        parsed = robust_json_parse(raw)
        
        if parsed and isinstance(parsed, dict) and "sentimiento" in parsed:
            print(f"[{i}] ✅ Success")
            success_count += 1
        else:
            print(f"[{i}] ❌ Failed: {raw[:50].replace(chr(10), ' ')}...")
            fail_count += 1

    print(f"\nStats: Success={success_count}, Fail={fail_count}")

if __name__ == "__main__":
    test_parsing()
