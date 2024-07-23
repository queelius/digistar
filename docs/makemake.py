import re

def generate_makefile(index_file, makefile_output):
    with open(index_file, 'r') as f:
        content = f.read()

    # Extract the list of markdown files from the index.md
    markdown_files = re.findall(r'\(([\w\-_]+\.md)\)', content)
    
    # Ensure the extracted list is not empty
    if not markdown_files:
        raise ValueError("No markdown files found in the index.md file")

    # Create the Makefile content
    makefile_content = "# Makefile to concatenate markdown files into a single document\n\n"
    makefile_content += "# List of markdown files in order\n"
    makefile_content += "MARKDOWN_FILES = \\\n"
    makefile_content += " \\\n    ".join(markdown_files) + "\n\n"

    makefile_content += """\
# Output file
OUTPUT_FILE = design_document.md

# Concatenate markdown files
$(OUTPUT_FILE): $(MARKDOWN_FILES)
\tcat $(MARKDOWN_FILES) > $(OUTPUT_FILE)

.PHONY: clean
clean:
\trm -f $(OUTPUT_FILE)
"""

    # Write the Makefile content to the output file
    with open(makefile_output, 'w') as f:
        f.write(makefile_content)

if __name__ == "__main__":
    index_file = 'index.md'
    makefile_output = 'Makefile'
    generate_makefile(index_file, makefile_output)
