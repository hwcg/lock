#!/usr/bin/env python3
"""Extract exact special token values from 1.md and write special_tokens.py."""

import re

with open('/root/dsv4/1.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the special_tokens.py code block
# Search for the section starting with the special_tokens.py header
pattern = r'### 1\.9 `src/deepseek_v4/tokenizer/special_tokens\.py`\s*```\n(.*?)```'
match = re.search(pattern, content, re.DOTALL)
if not match:
    print("ERROR: Could not find special_tokens.py code block")
    exit(1)

code = match.group(1)
# Remove leading/trailing whitespace but keep internal structure
code = code.strip() + '\n'

with open('/root/dsv4/src/deepseek_v4/tokenizer/special_tokens.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("special_tokens.py written successfully")
# Verify key tokens
for line in code.split('\n'):
    if line.startswith('THINK_START') or line.startswith('THINK_END') or line.startswith('TOOL_CALL_START') or line.startswith('TOOL_CALL_END') or line.startswith('TOOL_RESPONSE_START') or line.startswith('TOOL_RESPONSE_END'):
        print(f"  {line}")
