# Determining the most effective variant caller with machine learning

My project consists of 2 parts:

1. Train a machine learning classifier and predict the correct labels for a varaint caller
2. Compare model predicted labels and crowdsourced labels for the variant caller using: predict_proba and tSNE plots

#### Background
There is substantial disagreement between various variant calling pipelines (i.e.: a tools that tells a user whether a variant exists within a particular site within the genome)

There have been many efforts to resolve the discrepancy between vairant callers, but an effective way to resolve these issues has yet to be identified

Crowdsourcing efforts have been launched to more confidently identify variants within the human genome
For the following, I will use a machine learning classifier - Random Forest Classifier - to determine the accuracy of the genotype calls generated by a varaint caller built based on Illumina technology

***Overview***

Train the classifier using crowdsource data
Determine the accuracy of the classifier
Determine how well the predictions made by the classfier correlate with the calls made by the variant caller


#### Data Summary
Structural varaints from a Personal Genome Project genome (Ashkenazi Jewish son) were assessed using an Illumnia sequencing and variant calling pipeline. 

* 1516 Instances (Variants)
* 43 Features (Generated via the variant calling pipeline)

A study led by Peyton Greenside (Stanford) and Google Verily used crowdsourcing to assign genotypes to each of the 1516 variants. These genotypes will be compared to the genotypes generated by the variant calling pipeline. The crowdsourced gentoypes are considered 'ground truth' and will be used in the initial training/testing of the model.

#### Labels

**Crowdsource Data**

| Label | Definition           |
|-------|----------------------|
|   0   | Homozygous Variant   |
|   1   | Heterozygous Variant |
|   2   | Homozygous Reference |


**Variant Caller Data**

| Label | Definition           |
|-------|----------------------|
|   0   | Homozygous Reference |
|   1   | Heterozygous Variant |
|   2   | Homozygous Variant   |
|   -1  | Unknown              |





Crowdsource data source: http://biorxiv.org/content/early/2016/12/13/093526
