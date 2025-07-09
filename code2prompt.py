import os

def collect_python_files(root_dir, max_lines_per_file=1000):
    prompt_parts = []
    prompt_parts.append("以下是我的Python项目中的文件内容，请帮我理解和记住每个模块。后续我会基于此进行提问或修改建议。\n")

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.cu') or filename.endswith('.h') or filename.endswith('.cpp') or filename.endswith('.cuh'):
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, root_dir)
                with open(full_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                truncated = len(lines) > max_lines_per_file
                file_content = ''.join(lines[:max_lines_per_file])
                prompt_parts.append(f"---\n文件路径: {rel_path}\n内容:\n```python\n{file_content}\n```")
                if truncated:
                    prompt_parts.append(f"⚠️ 内容过长，仅展示前 {max_lines_per_file} 行。\n")

    return '\n'.join(prompt_parts)

# 使用示例
if __name__ == '__main__':
    project_path = "./csrc"  # 修改为你的项目路径
    result = collect_python_files(project_path)
    with open("prompt_output.txt", "w", encoding="utf-8") as out_file:
        out_file.write(result)
    print("✅ Prompt 已保存到 prompt_output.txt")
