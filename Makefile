.PHONY: all scripts render copy

SCRIPT_DIRS := \
	01_plasma_physics \
	02_envelope_equations \
	04_halo_formation

DOC_DIRS := \
	00_intro \
	01_plasma_physics \
	02_envelope_equations \
	03_current_limits \
	04_halo_formation \
	05_collisions

all: scripts render copy

scripts:
	for d in $(SCRIPT_DIRS); do \
		(cd $$d/scripts && ./run.sh); \
	done

render:
	quarto render

copy:
	rm -rf outputs
	mkdir -p outputs
	for d in $(DOC_DIRS); do \
		cp $$d/main.pdf outputs/$$d.pdf; \
		cp $$d/main.html outputs/$$d.html; \
	done