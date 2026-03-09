import argparse
import logging
import os

from flask import Flask, jsonify, render_template_string, request
from safetensors import safe_open

app = Flask(__name__)
logger = logging.getLogger(__name__)
MODEL_DIR = ""
SAFETENSORS_TREE_CACHE = {}

def build_safetensors_tree(file_path):
    """Build a nested tensor tree for one safetensors file."""
    root_node = {}
    with safe_open(file_path, framework="pt") as tensor_file:
        for tensor_name in tensor_file.keys():
            tensor_slice = tensor_file.get_slice(tensor_name)
            tensor_shape = list(tensor_slice.get_shape())
            tensor_dtype = str(tensor_slice.get_dtype()).replace("torch.", "")

            parts = tensor_name.split(".")
            current_node = root_node
            for part_index, part_name in enumerate(parts):
                if part_name not in current_node:
                    current_node[part_name] = {"children": {}, "is_leaf": False}

                if part_index == len(parts) - 1:
                    current_node[part_name]["is_leaf"] = True
                    current_node[part_name]["shape"] = tensor_shape
                    current_node[part_name]["dtype"] = tensor_dtype
                else:
                    current_node = current_node[part_name]["children"]
    return root_node


def build_file_entries(directory):
    """Build all file entries under the input directory."""
    file_entries = []
    for file_name in sorted(os.listdir(directory)):
        file_path = os.path.join(directory, file_name)
        if not os.path.isfile(file_path):
            continue

        if file_name.endswith(".safetensors"):
            file_entries.append(
                {
                    "file_name": file_name,
                    "file_type": "safetensors",
                }
            )
        else:
            file_entries.append(
                {
                    "file_name": file_name,
                    "file_type": "text",
                }
            )
    return file_entries

# --- Frontend HTML template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Model File Explorer</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; 
               background-color: #ffffff; color: #24292f; margin: 0; padding: 20px; }
        .layout { max-width: 1600px; margin: 0 auto; display: grid; grid-template-columns: 60% 40%; gap: 16px; }
        .container { border: 1px solid #d0d7de; border-radius: 6px; overflow: hidden; }
        .header { display: flex; background-color: #f6f8fa; border-bottom: 1px solid #d0d7de; 
                  padding: 10px 15px; font-weight: 600; font-size: 14px; color: #57606a; }
        .col-name { flex: 1; }
        .col-shape { width: 250px; text-align: left; }
        .col-dtype { width: 100px; text-align: left; }

        .node { border-bottom: 1px solid #f0f0f0; }
        .row { display: flex; align-items: center; padding: 6px 15px; transition: background 0.1s; }
        .row:hover { background-color: #f5f8ff; }

        .name-wrapper { flex: 1; display: flex; align-items: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .toggle-icon { width: 18px; height: 18px; display: inline-flex; align-items: center; justify-content: center; 
                       margin-right: 5px; color: #8c959f; font-size: 10px; transition: transform 0.2s; }

        .shape-val { width: 250px; font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace; 
                     font-size: 12px; color: #0969da; }
        .dtype-val { width: 100px; font-size: 12px; color: #57606a; font-weight: 500; }

        .children-container { display: none; margin-left: 20px; border-left: 1px solid #d0d7de; }
        .expanded > .children-container { display: block; }
        .expanded > .row .toggle-icon { transform: rotate(90deg); }

        .leaf .toggle-icon { visibility: hidden; }
        .dimmed { color: #8c959f; font-size: 12px; margin-left: 6px; }
        .clickable { cursor: pointer; }
        .file-row { background-color: #fbfcfe; }
        .error-text { color: #cf222e; }
        .text-panel-title { padding: 10px 15px; border-bottom: 1px solid #d0d7de; font-weight: 600; font-size: 14px; background-color: #f6f8fa; }
        .text-panel-content { margin: 0; padding: 12px 15px; white-space: pre-wrap; word-break: break-word; font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace; font-size: 12px; min-height: 500px; max-height: 70vh; overflow: auto; }
    </style>
</head>
<body>
    <h2 style="font-weight: 400; margin-bottom: 15px;">Model Files <small style="font-size: 14px; color: #666;">(Safetensors + Text Viewer)</small></h2>
    <div class="layout">
        <div class="container">
            <div class="header">
                <div class="col-name">File / Tensor Name</div>
                <div class="col-shape">Shape / Details</div>
                <div class="col-dtype">Type</div>
            </div>
            <div id="tree-root"></div>
        </div>
        <div class="container">
            <div class="text-panel-title" id="text-title">Text Preview</div>
            <pre class="text-panel-content" id="text-content">Click a text file on the left to preview its content.</pre>
        </div>
    </div>

    <script>
        async function init() {
            const response = await fetch('/api/files');
            const fileEntries = await response.json();
            const root = document.getElementById('tree-root');
            root.innerHTML = '';
            renderFiles(fileEntries, root);
        }

        function renderFiles(fileEntries, container) {
            fileEntries.forEach(fileEntry => {
                const fileNode = document.createElement('div');
                fileNode.className = 'node' + (fileEntry.file_type === 'safetensors' ? ' collapsed' : '');
                fileNode.dataset.isLoaded = 'false';

                const fileRow = document.createElement('div');
                fileRow.className = 'row file-row' + (fileEntry.file_type === 'text' ? ' clickable' : '');
                const detailText = fileEntry.file_type === 'safetensors' ? 'Click to load tensor tree' : 'Text preview';
                const typeText = fileEntry.file_type;
                fileRow.innerHTML = `
                    <div class="name-wrapper">
                        <span class="toggle-icon">${fileEntry.file_type === 'safetensors' ? '▶' : ''}</span>
                        <span style="color: #24292f">${fileEntry.file_name}</span>
                    </div>
                    <div class="shape-val">${detailText}</div>
                    <div class="dtype-val">${typeText}</div>
                `;
                fileNode.appendChild(fileRow);

                if (fileEntry.file_type === 'safetensors') {
                    const childrenBox = document.createElement('div');
                    childrenBox.className = 'children-container';
                    fileNode.appendChild(childrenBox);
                    fileRow.classList.add('clickable');
                    fileRow.onclick = async (event) => {
                        await loadSafetensorsTree(fileEntry.file_name, childrenBox, fileNode);
                        fileNode.classList.toggle('expanded');
                        event.stopPropagation();
                    };
                } else if (fileEntry.file_type === 'text') {
                    fileRow.onclick = async () => {
                        await loadTextFile(fileEntry.file_name);
                    };
                }

                container.appendChild(fileNode);
            });
        }

        async function loadSafetensorsTree(fileName, childrenContainer, fileNode) {
            if (fileNode.dataset.isLoaded === 'true') {
                return;
            }
            if (fileNode.dataset.isLoaded === 'loading') {
                return;
            }

            fileNode.dataset.isLoaded = 'loading';
            childrenContainer.innerHTML = '<div class="row"><div class="name-wrapper"><span class="dimmed">Loading tensor tree...</span></div></div>';

            const response = await fetch(`/api/safetensors_tree?file_name=${encodeURIComponent(fileName)}`);
            const payload = await response.json();
            if (payload.error) {
                childrenContainer.innerHTML = `<div class="row"><div class="name-wrapper"><span class="error-text">Error: ${payload.error}</span></div></div>`;
                fileNode.dataset.isLoaded = 'false';
                return;
            }

            childrenContainer.innerHTML = '';
            renderTensorNodes(payload.tensor_tree, childrenContainer);
            fileNode.dataset.isLoaded = 'true';
        }

        function renderTensorNodes(nodes, container, parentPath = '') {
            const sortedKeys = Object.keys(nodes).sort((a, b) => {
                return a.localeCompare(b, undefined, {numeric: true, sensitivity: 'base'});
            });

            sortedKeys.forEach(key => {
                const node = nodes[key];
                const currentPath = parentPath ? `${parentPath}.${key}` : key;
                const displayName = node.is_leaf ? currentPath : key;
                const nodeEl = document.createElement('div');
                nodeEl.className = 'node' + (node.is_leaf ? ' leaf' : ' collapsed');

                const row = document.createElement('div');
                row.className = 'row';

                // Count child nodes for folders in the tensor tree.
                const childCount = node.is_leaf ? '' : `(${Object.keys(node.children).length})`;

                row.innerHTML = `
                    <div class="name-wrapper">
                        <span class="toggle-icon">▶</span>
                        <span style="color: ${node.is_leaf ? '#24292f' : '#0969da'}">${displayName}</span>
                        <span class="dimmed">${childCount}</span>
                    </div>
                    <div class="shape-val">${node.shape ? '[' + node.shape.join(', ') + ']' : ''}</div>
                    <div class="dtype-val">${node.dtype || ''}</div>
                `;

                nodeEl.appendChild(row);

                if (!node.is_leaf) {
                    const childrenBox = document.createElement('div');
                    childrenBox.className = 'children-container';
                    renderTensorNodes(node.children, childrenBox, currentPath);
                    nodeEl.appendChild(childrenBox);

                    row.classList.add('clickable');
                    row.onclick = (event) => {
                        nodeEl.classList.toggle('expanded');
                        event.stopPropagation();
                    };
                }

                container.appendChild(nodeEl);
            });
        }

        async function loadTextFile(fileName) {
            const response = await fetch(`/api/text?file_name=${encodeURIComponent(fileName)}`);
            const payload = await response.json();
            const textTitle = document.getElementById('text-title');
            const textContent = document.getElementById('text-content');
            if (payload.error) {
                textTitle.textContent = `Text Preview - ${fileName}`;
                textContent.textContent = `Error: ${payload.error}`;
                return;
            }
            textTitle.textContent = `Text Preview - ${payload.file_name}`;
            textContent.textContent = payload.content;
        }

        init();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/files')
def get_files():
    file_entries = build_file_entries(MODEL_DIR)
    return jsonify(file_entries)


@app.route('/api/safetensors_tree')
def get_safetensors_tree():
    file_name = request.args.get("file_name", "")
    if not file_name:
        return jsonify({"error": "file_name is required"}), 400

    safe_file_name = os.path.basename(file_name)
    file_path = os.path.join(MODEL_DIR, safe_file_name)
    if not os.path.isfile(file_path):
        return jsonify({"error": "File not found"}), 404

    if not safe_file_name.endswith(".safetensors"):
        return jsonify({"error": "Only .safetensors files are supported"}), 400

    if safe_file_name in SAFETENSORS_TREE_CACHE:
        return jsonify({"file_name": safe_file_name, "tensor_tree": SAFETENSORS_TREE_CACHE[safe_file_name]})

    try:
        tensor_tree = build_safetensors_tree(file_path)
    except Exception as exception:
        logger.error("Failed to parse safetensors file %s: %s", safe_file_name, exception)
        return jsonify({"error": str(exception)}), 500

    SAFETENSORS_TREE_CACHE[safe_file_name] = tensor_tree
    return jsonify({"file_name": safe_file_name, "tensor_tree": tensor_tree})


@app.route('/api/text')
def get_text_file_content():
    file_name = request.args.get("file_name", "")
    if not file_name:
        return jsonify({"error": "file_name is required"}), 400

    safe_file_name = os.path.basename(file_name)
    file_path = os.path.join(MODEL_DIR, safe_file_name)
    if not os.path.isfile(file_path):
        return jsonify({"error": "File not found"}), 404

    if safe_file_name.endswith(".safetensors"):
        return jsonify({"error": "Use the safetensors tree view for this file"}), 400

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as file_handler:
            file_content = file_handler.read()
    except Exception as exception:
        return jsonify({"error": str(exception)}), 500

    return jsonify({"file_name": safe_file_name, "content": file_content})

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Start a local web server to inspect files in one directory.\n\n"
            "Behavior:\n"
            "  - Lists all regular files in the target directory.\n"
            "  - Loads .safetensors tensor trees lazily (only when you expand a file).\n"
            "  - Previews non-.safetensors files as plain text in the right panel.\n"
            "  - Uses UTF-8 text decoding with replacement for invalid bytes."
        ),
        epilog=(
            "Examples:\n"
            "  python watch_server.py /path/to/model_output\n"
            "  python watch_server.py /path/to/model_output --port 8080\n\n"
            "After startup:\n"
            "  - Open http://127.0.0.1:<port> in your browser.\n"
            "  - Click a .safetensors row to load and expand its tensor tree.\n"
            "  - Click a non-.safetensors file to preview its text content."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "folder",
        help=(
            "Directory path that contains model files. "
            "All regular files in this directory are listed in the UI."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="HTTP port for the local web server. Valid range: 1-65535. Default: 5000.",
    )
    parsed_arguments = parser.parse_args()

    if not os.path.isdir(parsed_arguments.folder):
        logger.error("%s is not a directory.", parsed_arguments.folder)
    else:
        MODEL_DIR = parsed_arguments.folder
        logger.info("Viewer running at http://127.0.0.1:%s", parsed_arguments.port)
        app.run(host='0.0.0.0', port=parsed_arguments.port)
