# Makefile for ph-project textbook overview
# Requires: lualatex, biber, latexmk

MAIN = overview

.PHONY: pdf clean

pdf:
	latexmk $(MAIN).tex

clean:
	latexmk -C $(MAIN).tex
	rm -f $(MAIN).bbl $(MAIN).run.xml $(MAIN).bcf
