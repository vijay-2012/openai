from nbconvert import PythonExporter
import nbformat

def export_ipynb_to_py(ipynb_file, output_file):
    # Load the IPython Notebook
    with open(ipynb_file, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)

    # Create a PythonExporter instance
    exporter = PythonExporter()

    # Export the notebook to a Python script
    python_code, _ = exporter.from_notebook_node(notebook)

    # Write the Python code to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(python_code)
