# Makefile to concatenate markdown files into a single document

# List of markdown files in order
MARKDOWN_FILES = \
    introduction.md \
    overview.md \
    big-atom-fundamentals.md \
    radiation.md \
    nuclear-big-atom-physics.md \
    black-holes.md \
    morse-potential.md \
    tensor-springs.md \
    big-atom-clusters.md \
    amorphous-objects.md \
    space-topology.md \
    data-management-and-spatial-index.md \
    grid-spatial-index.md \
    celestial-mechanics-dsl.md \
    message-passing-api.md \
    decentralized-space.md \
    agent-framework.md \
    base-units.md

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
