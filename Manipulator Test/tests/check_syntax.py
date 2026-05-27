
import os
import ast
import sys

def check_syntax(root_dir):
    print(f"Checking syntax for Python files in: {root_dir}")
    
    error_count = 0
    file_count = 0
    
    for root, dirs, files in os.walk(root_dir):
        if "site-packages" in root or ".git" in root or "__pycache__" in root:
            continue
            
        for file in files:
            if file.endswith(".py"):
                file_count += 1
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    # 1. Syntax Check
                    ast.parse(content, filename=file)
                    # print(f"  [OK] {file}")
                    
                except SyntaxError as e:
                    print(f"  [FAIL] {file}: {e}")
                    error_count += 1
                except Exception as e:
                    print(f"  [ERR] {file}: Could not read/parse ({e})")
                    error_count += 1
                    
    print("-" * 30)
    print(f"Checked {file_count} files.")
    if error_count == 0:
        print("PASS: No syntax errors found.")
    else:
        print(f"FAIL: Found {error_count} errors.")

if __name__ == "__main__":
    check_syntax(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
