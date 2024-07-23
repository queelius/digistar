# Makefile to concatenate markdown files into a single document

# List of markdown files in order
MARKDOWN_FILES = \
introduction.md \
    overview.md \
    big-atom-fundamentals.md \
    temperature-dynamics.md \
    nuclear-big-atom-physics.md \
    black-holes.md \
    morse-potential.md \
    tensor-springs.md \
    big-atom-clusters.md \
    space-topology.md \
    data-management-and-spatial-index.md \
    grid-spatial-index.md \
    celestial-mechanics-dsl.md \
    message-passing-api.md \
    decentralized-space.md \
    agent-framework.md

# Output file
OUTPUT_FILE = design_document.md

# Concatenate markdown files
$(OUTPUT_FILE): $(MARKDOWN_FILES)
	cat $(MARKDOWN_FILES) > $(OUTPUT_FILE)

.PHONY: clean
clean:
	rm -f $(OUTPUT_FILE)
