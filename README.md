# epi-express

Our solution to project 1 for the Machine Learning for Genomics course (263-5351-00L) offered at ETHZ

Goal is to predict gene expression from seven chromatin tracks: DNase accessibility plus H3K4me3, H3K4me1, H3K27ac, H3K36me3, H3K27me3 and H3K9me3

Trained on anonymous cell lines X1 and X2 on most autosomes, with chr14 and chr19 held out
for validation. Test set is chr1 of a third cell line, X3, meaning our solution should be able to generalize to unseen cell lines.

## Layout

model_infrastruture contains training, data_prep


## Inspo/Tutorial

https://github.com/aertslab/crested

## Tools 

https://github.com/deeptools/pyBigWig


## References

- The ENCODE Project Consortium. An integrated encyclopedia of DNA elements in the human genome. *Nature* 489, 57–74 (2012). https://doi.org/10.1038/nature11247
