# Resources for information
(tables including information on current drugs)
https://www.who.int/blueprint/priority-diseases/key-action/Table_of_therapeutics_Appendix_17022020.pdf?ua=1

(Websites on specific molecules)
https://zinc.docking.org/substances/ZINC000013915654/
http://zinc.docking.org/substances/ZINC000936069425/

(explanation on outbreak and identification of proteins related to the virus)
https://www.biorxiv.org/content/10.1101/2020.03.22.002386v3

(Great article on preventing/developing drugs to fight COVID-19)
https://www.sciencenews.org/article/coronavirus-covid19-repurposed-treatments-drugs

(score functions based on affinity classified in this paper)
https://www.preprints.org/manuscript/202002.0242/v1
# Software being used
-SwissADME (input the parent drug structure and it searches possible analogues) Docking

# Possible important information
(Youtube explanation of SwissADME)

(at 14:04 explains QSPR, a formula which can describe a molecule w/ numerical values relevant to its chemical and physical properties)

(at 17:13 explains classification models for PK-related behaviors, talks about inhibitors in SVM, also BOILED-egg (ML))

(21:06 the data is translated into SMILES, generate ellipse with Monte Carlo (MC))

(29:33 the SVM can be turn into a linear SVM and has binary outputs (Yes/No))

(32:44 talks about Brenk: problematic fragments like toxic, reactive, unstable)

(44:40 import molecules/add SMILES to SwissADME, it can take the name of the molecule purely and convert it to SMILES)

https://www.youtube.com/watch?v=6doVRePMHEg
(58:00 export results as a CSV, we might be able to process this with ML)

 (Passive gastro-intestinal (HIA) absorption and blood-brain barrier (BBB) permeation are predicted with the BOILED-Egg model) - great explanations on QSPR for SwissADME
 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5335600/
 
# Notes
-A docking experiment is where you have your target protein and your small molecule that you want to check whether it binds and how strongly.

-A derivative is a drug made from another drug

-Proteas cut proteins from long chains and drugs that are inhibitors limit the proteas of diseases similar to COVID-19

-Toxicity is the effect on something living (organism/tissue/cell). You must account for differences between populations (dosage of a rat vs a human)

-SwissADME is Ligand Based (will take a series of substances which have been successful drugs as a training set and determine a chemical which has more affinity)

