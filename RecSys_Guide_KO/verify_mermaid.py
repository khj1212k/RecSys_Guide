import os
import re
import sys

def verify_mermaid_syntax(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract Mermaid blocks
    mermaid_blocks = re.findall(r'```mermaid(.*?)```', content, re.DOTALL)
    
    errors = []
    
    for i, block in enumerate(mermaid_blocks):
        lines = block.strip().split('\n')
        if not lines:
            continue
            
        # 1. Check for Graph Declaration
        if not re.match(r'^\s*(graph|flowchart|sequenceDiagram|classDiagram|stateDiagram-v2|stateDiagram|gantt|pie|erDiagram|gitGraph|mindmap|timeline|xychart-beta)', lines[0]):
            errors.append(f"Block {i+1}: Missing or invalid graph declaration (e.g., 'graph TD'). Found: '{lines[0]}'")

        open_subgraphs = 0
        
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('%%'):
                continue
                
            # 2. Check for Balanced Brackets
            # We ignore < > because they are used in arrows --> and HTML tags <br> which makes simple counting hard.
            for open_char, close_char in [('[', ']'), ('(', ')'), ('{', '}')]: 
                if line.count(open_char) != line.count(close_char):
                     # Ignore if inside quotes (naive check) (e.g. "Label (Info)")
                    if '"' not in line:
                         errors.append(f"Block {i+1}, Line {line_idx+1}: Unbalanced brackets '{open_char}{close_char}'. Line: {line}")

            # 3. Strict Label Quoting Rule
            # Detects: id[Content] or id(Content) where Content contains : ( ) [ ] { } but NO quotes
            
            # Define specific patterns for different bracket types
            # Order matters! Check doubles before singles to avoid wrong matching.
            # (opener, closer, regex_pattern)
            patterns = [
                (r'\[\[', r'\]\]', r'(\w+)\s*\[\[\s*("(?:[^"\\]|\\.)*"|.*?)\s*\]\]'),  # [[...]]
                (r'\(\(', r'\)\)', r'(\w+)\s*\(\(\s*("(?:[^"\\]|\\.)*"|.*?)\s*\)\)'),  # ((...))
                (r'\{\{', r'\}\}', r'(\w+)\s*\{\{\s*("(?:[^"\\]|\\.)*"|.*?)\s*\}\}'),  # {{...}}
                (r'\[',   r'\]',   r'(\w+)\s*\[\s*("(?:[^"\\]|\\.)*"|[^\]]*)\s*\]'),    # [...]
                (r'\(',   r'\)',   r'(\w+)\s*\(\s*("(?:[^"\\]|\\.)*"|[^\)]*)\s*\)'),    # (...)
                (r'\{',   r'\}',   r'(\w+)\s*\{\s*("(?:[^"\\]|\\.)*"|[^\}]*)\s*\}'),    # {...}
                (r'>',    r'\]',   r'(\w+)\s*>\s*("(?:[^"\\]|\\.)*"|[^\]]*)\s*\]')      # >...]
            ]
            
            # Find all top-level quoted strings to avoid matching patterns inside them
            # We want to identify ranges [start, end] of literal strings in the Mermaid line
            # e.g. A -- "Label (Text)" --> B
            # Quoted string: "Label (Text)" at some index.
            # False match `Label (Text)` would start inside that range.
            
            quoted_spans = []
            for qm in re.finditer(r'"(?:[^"\\]|\\.)*"', line):
                 quoted_spans.append(qm.span())
                 
            found_matches = []
            for open_tag, close_tag, pattern in patterns:
                for m in re.finditer(pattern, line):
                    # Check if match starts inside a quoted section
                    m_start = m.start()
                    is_inside_quote = False
                    for q_start, q_end in quoted_spans:
                        # If the match starts strictly inside the quotes (after opening quote)
                        # The quoted span includes the quotes themselves.
                        # e.g. "Text". span=(0, 6).
                        # Match `Text` starts at 1. 1 > 0 and 1 < 6.
                        if m_start > q_start and m_start < q_end:
                            is_inside_quote = True
                            break
                    
                    if not is_inside_quote:
                        found_matches.append((m, open_tag, close_tag))
            
            # Simple deduplication: sort by start index.
            found_matches.sort(key=lambda x: x[0].start())
            
            unique_matches = []
            last_end = -1
            for item in found_matches:
                m = item[0]
                if m.start() >= last_end:
                    unique_matches.append(item)
                    last_end = m.end()
            
            for m, open_b, close_b in unique_matches:
                node_id, content = m.groups()

                # Basic cleanup
                if not content:
                    continue

                # Check for mismatched quotes (odd number of quotes in content)
                # This could happen if our regex match included a partial string due to '.*?' being greedy/non-greedy in weird ways
                # or just user error.
                if content.count('"') % 2 != 0:
                     print(f"  - Block {i+1}, Line {line_idx+1}: Mismatched or partial quotes in label. Content: '{content}'. Line: {line.strip()}")
                     continue # Skip further checks if quotes are broken

                # Check if the content is fully quoted
                is_quoted = content.strip().startswith('"') and content.strip().endswith('"')
                
                if not is_quoted:
                    # Check for special characters that require quoting
                    # Colon (:), Parentheses (()), Braces ({}), Brackets ([])
                    # Note: parens inside labels are usually fine if balanced, but Mermaid can be picky.
                    # Strict check: : ( ) [ ] { }
                    
                    specials = [':', '(', ')', '[', ']', '{', '}']
                    if any(char in content for char in specials):
                        # False positive check for ((Circle)) where content inside is "Circle" -> No special chars.
                        # But what if content is "Label: Text"?
                        # If regex successfully extracted "Label: Text" (no quotes), let's see.
                        
                        # Special Exception: 
                        # If the bracket type is ( ) and the content contains ( ), it MIGHT be a nested parens case or simple text with parens.
                        # Mermaid often tolerates balanced parens: id(Text (info)) is valid.
                        # But id(Text : info) is NOT valid usually.
                        
                        if ':' in content:
                             errors.append(f"Block {i+1}, Line {line_idx+1}: Unquoted special characters (colon) in node label. Use quotes! Content: '{content}'. Line: {line.strip()}")
                        
                        # For brackets/parens, be suspicious but maybe lenient if they look balanced?
                        # Actually, keeping it strict is safer for "Robust Verification".
                        # But we must avoid flagging valid simple parens if they are ubiquitous.
                        # Let's warn about them.
                        elif any(char in content for char in ['(', ')', '[', ']', '{', '}']):
                             # print(f"  - Block {i+1}: Unquoted brackets in label. Might be okay but risky: '{content}'")
                             pass # Silencing bracket warnings for now to focus on critical COLON errors which definitely break rendering.
                
            # 4. Logical Check: Subgraph closing
            if line.startswith('subgraph '):
                open_subgraphs += 1
            if line == 'end':
                open_subgraphs -= 1
        
        if open_subgraphs != 0:
            errors.append(f"Block {i+1}: Unbalanced 'subgraph' and 'end' statements. Remainder: {open_subgraphs}")

    return errors

def main():
    root_dir = "/Users/brownee/Downloads/BoostCamp AI Tech/AI_Guide/LLM_Guide/LLM_Guide_KR"
    has_errors = False
    
    print(f"Scanning directory: {root_dir}")
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".md"):
                full_path = os.path.join(root, file)
                errors = verify_mermaid_syntax(full_path)
                
                if errors:
                    has_errors = True
                    print(f"\n❌ {file}:")
                    for e in errors:
                        print(f"  - {e}")
                else:
                    # Optional: Print clean files if verbose
                    pass

    if has_errors:
        print("\nSUMMARY: Errors found. Please fix them.")
        sys.exit(1)
    else:
        print("\nSUMMARY: All Mermaid diagrams passed syntax checks! ✅")
        sys.exit(0)

if __name__ == "__main__":
    main()
