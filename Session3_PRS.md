# Practice Session #3: Polygenic Risk Score (May 1, 2025)

In this session, we are going to construct polygenic risk score using PRS-CS. \
References : [PRS-CS github](https://github.com/getian107/PRScs), [PRS-CS paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6467998/). \
The data we are going to use are already preprocessed or downloaded.

### 1. Log in to your account and access the desired compute node
``` 
ssh YOURID@147.47.200.131 -p 22555
```
``` 
ssh leelabsg15
```


The server already has Python built-in, so there’s no need to activate a Conda environment. 

### 2. Navigate to the 3_PRS folder and create a new directory called result to store outputs from the practice session
```
cd 3_PRS
``` 
```
mkdir result 
``` 

### 3. While staying in the 3_PRS directory, run the PRS-CS to calculate polygenic risk scores.
```
python PRScs/PRScs.py \
--ref_dir=data/reference/ldblk_1kg_eas \
--bim_prefix=data/plink/sample \
--sst_file=data/summary_stat/sumstats_prscs.txt \
--n_gwas=177618 \
--out_dir=result/prscs
```
Use the nohup command to run PRS-CS in the background, allowing the process to continue even if the terminal is closed. The output will be saved to result/nohup.out.
```
nohup python PRScs/PRScs.py \
--ref_dir=data/reference/ldblk_1kg_eas \
--bim_prefix=data/plink/sample \
--sst_file=data/summary_stat/sumstats_prscs.txt \
--n_gwas=177618 \
--out_dir=result/prscs > result/nohup.out &

``` 

### 4. Navigate to the result folder and combine PRS-CS Output Files (chr1–chr22) into a Single File
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
/data/GCDA/plink \
--bfile data/plink/sample \
--score result/prscs_chr1-22.txt 2 4 6 \
--out result/score
``` 
#### NOTE: Since are using relative paths in the commands (except for Step 4), please make sure to run all commands from within the 3_PRS directory. Alternatively, you can modify the commands to use absolute paths if you’re running them from a different location.
