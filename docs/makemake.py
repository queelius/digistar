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
# Output files
OUTPUT_MD = design_document.md
OUTPUT_PDF = design_document.pdf

# Default target
all: $(OUTPUT_MD)

# Concatenate markdown files
$(OUTPUT_MD): $(MARKDOWN_FILES)
    cat $(MARKDOWN_FILES) > $(OUTPUT_MD)

# Generate PDF from markdown
$(OUTPUT_PDF): $(OUTPUT_MD)
    pandoc $(OUTPUT_MD) -o $(OUTPUT_PDF)

# Clean up generated files
.PHONY: clean
clean:
    rm -f $(OUTPUT_MD) $(OUTPUT_PDF)
"""

    # Write the Makefile content to the output file
    with open(makefile_output, 'w') as f:
        f.write(makefile_content)

if __name__ == "__main__":
    index_file = 'index.md'
    makefile_output = 'Makefile'
    generate_makefile(index_file, makefile_output)
