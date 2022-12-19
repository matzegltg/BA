FILE=main
PDFLATEXARGS=-synctex=1 -file-line-error -halt-on-error
LUALATEXARGS=--syntex=1 --file-line-error --halt-on-error
# lualatex, pdflatex, ...
COMPILER=pdflatex
COMPILERARGS=$(LUALATEXARGS)

pdf:
	$(COMPILER) $(COMPILERARGS) $(FILE).tex
	bibtex $(FILE)
	$(COMPILER) $(COMPILERARGS) $(FILE).tex
	$(COMPILER) $(COMPILERARGS) $(FILE).tex

clean:
	rm -f *.aux $(FILE).bbl $(FILE).lof *.log $(FILE).lot $(FILE).toc $(FILE).synctex.gz $(FILE).blg $(FILE).lol $(FILE).out images/*-eps-converted-to.pdf
