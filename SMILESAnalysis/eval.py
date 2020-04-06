import csv
import textdistance as td

#opening input file Candidates.csv
o = open('Candidates.csv')
l = list(o.readlines())
filereader = csv.reader(o, delimiter='	')
s = []

#ignores the first line of input file
for row in l[1:]:
	s.append(row.split(','))
	
for i in s:
	i[-1] = i[-1][:-1]

#initialising string constants
atp = "C1=NC(=C2C(=N1)N(C=N2)[C@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O)N"
nl = "\n"
#Names of similarity*/difference^ measures used
dl = "Damerau-Levenshtein^: "
lev = "Levenshtein^: "
over = "Overlap*: "
lcsseq = "Longest Common Subsequence*: "
lcsstr = "Longest Common Substring*: "
gest = "Gestalt Pattern Matching*: "


#output file
res = open('results.txt','w')

#write to outpute file, for each line of the input file *except the first line*
for i in s:
    smiles = i[1]
    name = i[0]
    lns = []
    diff = abs(len(smiles)-len(atp))
    lns.append(name)
    lns.append(nl)
    lns.append(i[-1])
    lns.append(nl)
    lns.append(dl)
    lns.append(str(td.damerau_levenshtein(smiles, atp)))
    lns.append(nl)
    lns.append(lev)
    lns.append(str(td.levenshtein(smiles, atp)))
    lns.append(nl)
    lns.append(over)
    lns.append(str(td.overlap(smiles, atp)))
    lns.append(nl)
    lns.append(lcsseq)
    lns.append(str(len(td.lcsseq(smiles, atp))))
    lns.append(nl)
    lns.append(lcsstr)
    lns.append(str(len(td.lcsstr(smiles, atp))))
    lns.append(nl)
    lns.append(gest)
    lns.append(str(td.ratcliff_obershelp(smiles, atp)))
    lns.append(nl)
    lns.append(nl)
    lns.append('________________________________')
    lns.append(nl)
    res.writelines(lns)
    

res.close()


    
    
