#!/usr/bin/env python3
"""Extract all tokenizer code from 1.md and write to the correct files."""
import re
import os

with open('/root/dsv4/1.md', 'r', encoding='utf-8') as f:
    content = f.read()

def extract_code(section_header, content):
    """Extract code block following a section header."""
    # Pattern: ### X.Y `path` followed by ```\ncode\n```
    pattern = rf'{re.escape(section_header)}\s*```\n(.*?)```'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip() + '\n'
    print(f"WARNING: Could not find section '{section_header}'")
    return None

files_to_extract = {
    '### 1.9 `src/deepseek_v4/tokenizer/special_tokens.py`': '/root/dsv4/src/deepseek_v4/tokenizer/special_tokens.py',
    '### 1.10 `src/deepseek_v4/tokenizer/bpe.py`': '/root/dsv4/src/deepseek_v4/tokenizer/bpe.py',
    '### 1.11 `src/deepseek_v4/tokenizer/encoding.py`': '/root/dsv4/src/deepseek_v4/tokenizer/encoding.py',
    '### 1.12 `src/deepseek_v4/tokenizer/tokenizer.py`': '/root/dsv4/src/deepseek_v4/tokenizer/tokenizer.py',
    '### 1.13 `src/deepseek_v4/tokenizer/trainer.py`': '/root/dsv4/src/deepseek_v4/tokenizer/trainer.py',
    '### 1.14 `scripts/train_tokenizer.py`': '/root/dsv4/scripts/train_tokenizer.py',
    '### 1.16 `tests/test_tokenizer.py`': '/root/dsv4/tests/test_tokenizer.py',
}

# Also write the __init__.py manually since it doesn't have a typical code block
init_content = '''"""DeepSeek-V4 分词器子包。"""

from deepseek_v4.tokenizer.special_tokens import SpecialTokens, ALL_SPECIAL_TOKENS
from deepseek_v4.tokenizer.tokenizer import DeepseekV4Tokenizer
from deepseek_v4.tokenizer.bpe import BPETokenizer, BPETrainer
from deepseek_v4.tokenizer import encoding

__all__ = [
    "SpecialTokens",
    "ALL_SPECIAL_TOKENS",
    "DeepseekV4Tokenizer",
    "BPETokenizer",
    "BPETrainer",
    "encoding",
]
'''

# Ensure directories exist
for path in files_to_extract.values():
    os.makedirs(os.path.dirname(path), exist_ok=True)

# Write __init__.py
with open('/root/dsv4/src/deepseek_v4/tokenizer/__init__.py', 'w', encoding='utf-8') as f:
    f.write(init_content)
print("Written: /root/dsv4/src/deepseek_v4/tokenizer/__init__.py")

# Extract and write each file
for header, path in files_to_extract.items():
    code = extract_code(header, content)
    if code:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(code)
        print(f"Written: {path}")

        # Verify special tokens
        if 'special_tokens.py' in path:
            for line in code.split('\n'):
                if line.startswith('THINK_') or line.startswith('TOOL_CALL_') or line.startswith('TOOL_RESPONSE_'):
                    print(f"  {line}")
    else:
        print(f"FAILED: {path}")

print("\nDone!")
