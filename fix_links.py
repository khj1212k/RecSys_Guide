import os
import re

ROOT_DIRS = ["RecSys_Guide_EN", "RecSys_Guide_KO"]

NAV_STRUCTURE = [
    ("README.md", "Home/홈"),
    ("01_Traditional_Models/README.md", "01. Traditional Models/01. 전통적 모델"),
    ("01_Traditional_Models/01_Collaborative_Filtering/README.md", "Collaborative Filtering/협업 필터링"),
    ("01_Traditional_Models/01_Collaborative_Filtering/01_Memory_Based/README.md", "Memory-based/메모리 기반"),
    ("01_Traditional_Models/01_Collaborative_Filtering/02_Model_Based/README.md", "Model-based/모델 기반"),
    ("01_Traditional_Models/02_Content_Based_Filtering/README.md", "Content-based Filtering/콘텐츠 기반 필터링"),
    ("02_Machine_Learning_Era/README.md", "02. Machine Learning Era/02. 과도기 및 통계적 모델"),
    ("03_Deep_Learning_Era/README.md", "03. Deep Learning Era/03. 딥러닝 기반 모델"),
    ("03_Deep_Learning_Era/01_MLP_Based/README.md", "MLP-based/MLP 기반"),
    ("03_Deep_Learning_Era/02_Sequence_Session_Based/README.md", "Sequence/Session-based/순차/세션 기반"),
    ("03_Deep_Learning_Era/03_Graph_Based/README.md", "Graph-based/그래프 기반"),
    ("03_Deep_Learning_Era/04_AutoEncoder_Based/README.md", "AutoEncoder-based/오토인코더 기반"),
    ("04_SOTA_GenAI/README.md", "04. SOTA & GenAI/04. 최신 및 생성형 모델"),
    ("04_SOTA_GenAI/01_LLM_Based/README.md", "LLM-based/LLM 기반"),
]

# Defines the full navigation block template structure
# We will dynamically generate the links based on relative depth
def generate_nav_block(lang, rel_root):
    is_en = (lang == "EN")
    
    # Helper to format link
    def link(path, text_en, text_ko):
        display = text_en if is_en else text_ko
        return f"[{display}]({rel_root}{path})"

    # Structure
    # Using a fixed layout similar to what was requested
    # We will just generate the breakdown hardcoded because the indents matter
    
    home = link("README.md", "Home", "홈")
    head1 = link("01_Traditional_Models/README.md", "01. Traditional Models", "01. 전통적 모델")
    head2 = link("02_Machine_Learning_Era/README.md", "02. Machine Learning Era", "02. 과도기 및 통계적 모델")
    head3 = link("03_Deep_Learning_Era/README.md", "03. Deep Learning Era", "03. 딥러닝 기반 모델")
    head4 = link("04_SOTA_GenAI/README.md", "04. SOTA & GenAI", "04. 최신 및 생성형 모델")

    # Depth 2 & 3 links
    # Traditional
    cf = link("01_Traditional_Models/01_Collaborative_Filtering/README.md", "Collaborative Filtering", "협업 필터링")
    cbf = link("01_Traditional_Models/02_Content_Based_Filtering/README.md", "Content-based Filtering", "콘텐츠 기반 필터링")
    
    mem = link("01_Traditional_Models/01_Collaborative_Filtering/01_Memory_Based/README.md", "Memory-based", "메모리 기반")
    mod = link("01_Traditional_Models/01_Collaborative_Filtering/02_Model_Based/README.md", "Model-based", "모델 기반")
    
    # ML Era
    # Hybrid, FM are usually direct links or folders
    # Let's link to the folders if they exist as READMEs
    
    # Deep Learning
    mlp = link("03_Deep_Learning_Era/01_MLP_Based/README.md", "MLP-based", "MLP 기반")
    seq = link("03_Deep_Learning_Era/02_Sequence_Session_Based/README.md", "Sequence/Session-based", "순차/세션 기반")
    graph = link("03_Deep_Learning_Era/03_Graph_Based/README.md", "Graph-based", "그래프 기반")
    ae = link("03_Deep_Learning_Era/04_AutoEncoder_Based/README.md", "AutoEncoder-based", "오토인코더 기반")

    # SOTA
    llm = link("04_SOTA_GenAI/01_LLM_Based/README.md", "LLM-based", "LLM 기반")
    multi = link("04_SOTA_GenAI/02_Multimodal_RS.md", "Multimodal RS", "멀티모달 추천")
    gen = link("04_SOTA_GenAI/03_Generative_RS.md", "Generative RS", "생성형 추천")

    # Special Leaf Links for specific sections (hardcoded for simplicity in this script)
    # Actually, to make it robust, let's just use the main category links mostly, 
    # but the user requested "Global Navigation" which showed expanded tree.
    # I will replicate the expanded tree structure using the relative paths.
    
    # Helper for leaf
    def leaf(path, text_en, text_ko):
        return link(path, text_en, text_ko)

    # Simplified Tree for Navigation Block
    lines = []
    summary = "Global Navigation" if is_en else "전체 탐색 (RecSys 가이드)"
    lines.append(f"<details>")
    lines.append(f"<summary><strong>{summary}</strong></summary>")
    lines.append("")
    lines.append(f"- {home}")
    lines.append(f"- {head1}")
    lines.append(f"    - {cf}")
    lines.append(f"        - {mem}") # Shorten for brevity if needed? No, user wants links.
    # Actually, adding TOO many links makes it huge. 
    # But I should stick to the structure I used in the manual edits.
    # In manual edits I included leaves like 'User-based CF'.
    # Let's stick to a robust skeleton: Categories + Subcategories.
    lines.append(f"        - {mod}")
    lines.append(f"    - {cbf}")
    
    lines.append(f"- {head2}")
    # ML Era leaves
    # lines.append(f"    - ...") 
    
    lines.append(f"- {head3}")
    lines.append(f"    - {mlp}")
    lines.append(f"    - {seq}")
    lines.append(f"    - {graph}")
    lines.append(f"    - {ae}")
    
    lines.append(f"- {head4}")
    lines.append(f"    - {llm}")
    lines.append(f"    - {multi}")
    lines.append(f"    - {gen}")
    
    lines.append("</details>")
    return "\n".join(lines)


def get_up_link(lang, current_path_rel_to_root):
    # current_path_rel_to_root: e.g. "01_Traditional_Models/README.md"
    parts = current_path_rel_to_root.split('/')
    depth = len(parts) - 1 # README.md is depth 0 in folder
    
    # If it is the Root README
    if current_path_rel_to_root == "README.md":
        return None 

    # Logic:
    # If filename is README.md, its parent is the folder above.
    #   path: 01/README.md -> Parent is Root README (../README.md)
    #   path: 01/01/README.md -> Parent is 01/README.md (../README.md)
    # If filename is NOT README, its parent is the README in same folder (README.md) OR parent folder?
    # Usually "Up" means "Parent Directory Index".
    # For file 01/01/File.md -> Up is 01/01/README.md (which is ./README.md or just README.md)
    
    filename = parts[-1]
    
    if filename == "README.md":
        # Up goes to ../README.md
        target = "../README.md"
    else:
        # Up goes to README.md in current dir
        target = "README.md"
        
    text = "Up to Parent" if lang == "EN" else "상위 폴더로 이동"
    return f"[{text}<]({target})" # Arrow syntax: [< Up to Parent](...)


def fix_file(file_path, root_dir):
    rel_path = os.path.relpath(file_path, root_dir)
    # depth calculation
    # e.g. README.md -> depth 0. rel_root = ./ 
    # e.g. 01/README.md -> depth 1. rel_root = ../
    # e.g. 01/01/README.md -> depth 2. rel_root = ../../
    # e.g. 01/01/File.md -> depth 2 (file in depth 2 folder). rel_root = ../../
    
    # Actually, os.path.sep count is better
    # RecSys_Guide_EN/README.md -> rel_path = "README.md". count('/') = 0.
    # RecSys_Guide_EN/01/README.md -> rel_path = "01/README.md". count = 1.
    
    depth = rel_path.count(os.path.sep)
    if depth == 0:
        rel_root = "./" # Self
    else:
        rel_root = "../" * depth

    lang = "EN" if "RecSys_Guide_EN" in root_dir else "KO"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Generate Blocks
    nav_block = generate_nav_block(lang, rel_root)
    # Up Link (Only for non-root README)
    up_link = None
    if rel_path != "README.md":
        # logic:
        # README.md in subfolder -> ../README.md
        # File.md in subfolder -> README.md (sibling)
        
        # Wait, my generate_nav_block logic for link(path) was:
        # rel_root + path.
        # e.g. if I am in 01/README.md (depth 1), rel_root is ../
        # Home link = ../README.md. Correct.
        
        # Up link logic:
        # file: 01/README.md. Parent: Root. Target: ../README.md
        # file: 01/01/File.md. Parent: 01/01/README.md. Target: README.md
        
        parts = rel_path.split(os.path.sep)
        filename = parts[-1]
        
        arrow = "<"
        text = "Up to Parent" if lang == "EN" else "상위 폴더로 이동"
        
        if filename == "README.md":
            target = "../README.md"
        else:
            target = "README.md"
            
        up_link = f"[{arrow} {text}]({target})"

    # REPLACE CONTENT
    
    # 1. Remove existing Nav block (search for <details>...Global Navigation...</details>)
    # Using regex strictly might be hard if content varies. 
    # But usually it's at the top.
    
    # Strategy: 
    # If the file starts with a link or details, strip them until the first Header (#).
    # Then prepend the NEW Up link and Nav block.
    
    # Split by lines
    lines = content.splitlines()
    
    # Find the index of the first line starting with '#'
    head_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("# "): # Main Title
            head_idx = i
            break
            
    if head_idx == -1:
        print(f"Skipping {rel_path} (No title found)")
        return

    # Keep everything from the title onwards
    body = lines[head_idx:]
    
    # Construct new header
    new_header = []
    if up_link:
        new_header.append(up_link)
        new_header.append("")
    
    if rel_path != "README.md": # Root README might not need Up link, but needs Nav? 
        # Actually Root README usually has the Nav.
        # But 'Up Link' is None for Root.
        pass

    new_header.append(nav_block)
    new_header.append("")
    
    new_content = "\n".join(new_header + body)
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"Fixed {rel_path}")

def main():
    base_cwd = os.getcwd()
    for root_dir in ROOT_DIRS:
        full_root = os.path.join(base_cwd, root_dir)
        if not os.path.exists(full_root):
            print(f"Root {root_dir} not found")
            continue
            
        for path, dirs, files in os.walk(full_root):
            for file in files:
                if file.endswith(".md"):
                    fix_file(os.path.join(path, file), full_root)

if __name__ == "__main__":
    main()
