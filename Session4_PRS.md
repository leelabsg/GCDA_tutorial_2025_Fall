# Practice Session #3: Polygenic Risk Score 

In this session, we are going to construct polygenic risk score using PRS-CS. \
References : [PRS-CS github](https://github.com/getian107/PRScs), [PRS-CS paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6467998/). \
The data we are going to use are already preprocessed or downloaded.

### 0. Log in to your accound
``` 
ssh YOURID@147.47.200.131 -p 22555
```

### The server already has Python built-in, so there’s no need to activate a Conda environment. You will see 


### 1. Navigate to the 3_PRS folder and create a new directory called result to store outputs from the practice session
```
cd 3_PRS
``` 
```
mkdir result 
``` 

### 2. While staying in the 3_PRS directory, run the PRScs.py script to calculate polygenic risk scores using PRS-CS.
```
python PRScs/PRScs.py \
--ref_dir=data/reference/ldblk_1kg_eas \
--bim_prefix=data/plink/sample \
--sst_file=data/summary_stat/sumstats_prscs.txt \
--n_gwas=177618 \
--out_dir=result/prscs
``` 

### 3. Merge chr1 - chr22 beta files into one file 
```
cd result
``` 
```
for i in {1..22}; do cat "prscs_pst_eff_a1_b0.5_phiauto_chr$i.txt" >> prscs_chr1-22.txt; done
``` 

### 5. Move back to the parent directory and run PLINK to calculate the polygenic risk score (PRS) based on the output from PRS-CS.
Columns 2, 4, and 6 represent the SNP ID, the effect allele, and the effect size of the effect allele, respectively.

```
cd ..
```
```
/data/home/leelabguest/utils/plink \
--bfile data/plink/sample \
--score result/prscs_chr1-22.txt 2 4 6 \
--out result/score
``` 
