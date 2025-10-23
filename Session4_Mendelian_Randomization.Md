# Practice Session #4: Two-sample Mandelian Randomization (Oct 28, 2025)
In this session, we will learn how to perform two Sample MR with publicly available GWAS datasets. \
The overall process in this session is mainly based on the [TwoSampleMR in R vignette](https://mrcieu.github.io/TwoSampleMR/).

#### Two Sample MR

### 0. Installing twoSampleMR in R using Conda 
To avoid conflicts in dependecies, we will create new environment for two sample MR (twoMR) in this session.
First, lets install R and devtools in new conda environment:
```
conda create -n twoMR r-base r-essentials r-devtools r-remotes r-gmp -c conda-forge -c R 
conda activate twoMR ; R --no-save
```
Install TwoSampleMR pacakge from github: 
```
remotes::install_github("MRCIEU/TwoSampleMR")
```
Load the necessary packages:
```
library(TwoSampleMR)
```


### 1. Two Sample Mandelian Randomization Overview
The workflow for performing MR is as follows:

#### a. Select instruments for the exposure (perform LD clumping if necessary)
#### b. Extract the instruments from the IEU GWAS database for the outcomes of interest
- Or instead, you can either
    1) Create your own GWAS summary statistics
    2) Find the correct GWAS summary statistics for your MR design 
#### c. Harmonise the effect sizes for the instruments on the exposures and the outcomes to be each for the same reference allele
#### d. Perform MR analysis, sensitivity analyses, create plots, compile reports

![Twosamplemr_Diagram](https://user-images.githubusercontent.com/22064612/235571598-61784c94-bb0e-44ce-887d-c73a0593dbe6.png)


### 2. IEU GWAS database
The [IEU Open GWAS project](https://opengwas.io/datasets/) contains complete GWAS summary statistics from a large number of studies. You can browse them through the website. 

To use the API provided by IEU, you must log in with your school account, apply for a JSON Web Token (JWT), and then register the generated token. \
All necessary steps for this process can be found in the [API Tutorial](https://api.opengwas.io/).

From the package, available_outcomes() will list up outcomes obtainable from IEU.
```
ao <- available_outcomes()
head(ao)
```


### 3. Exposure Data Example
In this part of session, we will follow the example from the package using formatted summary statistics. \
We want to discover causal relationship between BMI and CHD. \
BMI comes from [GIANT](https://gwas.mrcieu.ac.uk/datasets/ieu-a-2/) Consortium \
CHD is from [CARDIoGRAMplusC4D](https://gwas.mrcieu.ac.uk/datasets/ieu-a-7/). 

You can import exposure data using the IEU API. 
```
bmi_exp_dat <- extract_instruments(outcomes = 'ieu-a-2')
head(bmi_exp_dat)
```


(Optional) You can import your own GWAS summary for exposure data. \
Example:
```
bmi_file <- system.file("extdata/bmi.csv", package = "TwoSampleMR")

bmi_exp_dat2 <- read_exposure_data(
    filename = bmi_file,
    sep = ",",
    snp_col = "rsid",
    beta_col = "effect",
    se_col = "SE",
    effect_allele_col = "a1",
    other_allele_col = "a2",
    eaf_col = "a1_freq",
    pval_col = "p-value",
    units_col = "Units",
    gene_col = "Gene",
    samplesize_col = "n"
)

head(bmi_exp_dat2)
```
For standard two sample MR, it is important to ensure that the **instruments for the exposure are independent**. If you are going to extract exposure variables from GWAS summary of your own or other studies, it is necessary to perform **clumping** to ensure independence between instrumental variables. 

For LD clumpig, you can use [clump_data() function](https://mrcieu.github.io/TwoSampleMR/reference/clump_data.html) provided by the "TwoSampleMR" package. 

### 4. Outcome Data Example
Following similar procedure as Exposure data acquirment, let's bring CHD summary statistics. By taking snps argument, we can scope the common variant list with BMI
```
chd_out_dat <- extract_outcome_data(
    snps = bmi_exp_dat$SNP,
    outcomes = 'ieu-a-7'
)
head(chd_out_dat)
```
(Optional) You can import your own GWAS summary file for outcome. [GWAS catalog](https://www.ebi.ac.uk/gwas/) provides some useful GWAS summary data.
```
chd_out_dat2 <- read_outcome_data(
    snps = bmi_exp_dat$SNP,
    filename = "/data/GCDA_2025_2/4_MR/26343387-GCST003116-EFO_0000378-build37.f.tsv",
    sep = "\t",
    snp_col = "variant_id",
    beta_col = "beta",
    se_col = "standard_error",
    effect_allele_col = "effect_allele",
    other_allele_col = "other_allele",
    eaf_col = "effect_allele_frequency",
    pval_col = "p_value",
    samplesize_col = "nstudy"
)
head(chd_out_dat2)
```


### 5. Harmonizing Data
Harmonizing data step is required because there are mismatches between alleles in exposure and outcome. Recent GWASs typically present the effects of a SNP in reference to the allele on the forward strand but not necessarily.

```
dat <- harmonise_data(
    exposure_dat = bmi_exp_dat, 
    outcome_dat = chd_out_dat
)
```
There are three options to harmonising the data.

1. Assume all alleles are presented on the forward strand 
2. Try to infer the forward strand alleles using allele frequency information 
3. Correct the strand for non-palindromic SNPs, but drop all palindromic SNPs \
By default, the harmonise_data function uses option 2, but this can be modified using the action argument, e.g. harmonise_data(exposure_dat, outcome_dat, action = 3).


### 6. Perform MR
Before we perform MR, lets find out which method is available in the package
```
mr_method_list()
```
Let's try MR with Egger, weighted median, and IVW. You can add list of methods or methods set by default.

```
res <- mr(dat, method_list = c("mr_egger_regression","mr_weighted_median","mr_ivw"))
res
```
Some Methods can perform sensitivity test which are implemented in mr_heterogeneity()
```
mr_heterogeneity(dat)
```
Intercept term in MR egger regression indicate the presence of Horizontal Pleiotropy.
```
mr_pleiotropy_test(dat)
```

Visualizing MR analysis with plot can be performed as below:
Let's save the figure using ggsave() from ggplot2
```
library(ggplot2)

p1 <- mr_scatter_plot(res, dat)
ggplot2::ggsave(p1[[1]], file = <FILE_PATH>, width = 7, height = 7)
```
In case that you want to download the plot to local computer, you can use the script below on your local terminal.
```
scp -P 22555 -J <USER_ID>@147.47.200.131 <USER_ID>@leelabsg11:<FROM_FILE_PATH> <TO_FILE_PATH>
```
![filename](https://user-images.githubusercontent.com/22064612/235652335-b57f0381-b820-47e0-a07f-da09564f9c0b.png)


Single SNP MR can be also performed by following. We can visualize the results via forest and funnel plots
```
res_single <- mr_singlesnp(dat)
head(res_single)

# forest plot
p2 <- mr_forest_plot(res_single)
ggplot2::ggsave(p2[[1]], file = <FILE_PATH>, width = 7, height = 7)

# funnel plot
p3 <- mr_funnel_plot(res_single)
ggplot2::ggsave(p3[[1]], file = <FILE_PATH>, width = 7, height = 7)
```
To see if a single snp is driving the association, we can perform leave-one-out MR.
In this case, we can see that there is no single variant that driving the association between exposure and outcome.
```
res_loo <- mr_leaveoneout(dat)
p4 <- mr_leaveoneout_plot(res_loo)
ggplot2::ggsave(p4[[1]], file = <FILE_PATH>, width = 7, height = 7)
```



