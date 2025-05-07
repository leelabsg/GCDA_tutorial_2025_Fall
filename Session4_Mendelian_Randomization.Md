# Practice Session #4: Two-sample Mandelian Randomization (May 8, 2025)
In this session, we will learn how to perform two Sample MR with publicly available GWAS datasets. \
The overall process in this session is mainly based on the [TwoSampleMR in R vignette](https://mrcieu.github.io/TwoSampleMR/).

#### Two Sample MR

### 0. Installing twoSampleMR in R using Conda 
To avoid conflicts in dependecies, we will create new environment for two sample MR (twoMR) in this session.
First, lets install R and devtools in new conda environment (Already installed as twoMR under leelab server)
```
conda create -n twoMR r-base r-essentials r-devtools r-remotes r-gmp -c conda-forge -c R 
conda activate twoMR ; R --no-save
```
TwoSampleMR pacakge is installed from github 
```
remotes::install_github("MRCIEU/TwoSampleMR")
```
Next, Load the necessary packages
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
The [IEU](https://gwas.mrcieu.ac.uk/) GWAS database (IGD) contains complete GWAS summary statistics from a large number of studies. You can browse them through the website. \
<img width="746" alt="image" src="https://user-images.githubusercontent.com/22064612/235642892-b283511b-6e2e-410a-9f7b-3c0611b325f7.png">

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

(Optional) You can import exposure data using the IEU API.
Example
```
bmi_exp_dat <- extract_instruments(outcomes = 'ieu-a-2')
```

In this practice, we are going to use "MRInstruments" library to import instrumental variables for MR.
```
remotes::install_github("MRCIEU/MRInstruments")
library(MRInstruments)
data(gwas_catalog)
```
```
bmi_gwas <-
  subset(gwas_catalog,
         PubmedID == 25673413)
bmi_exp_dat <- format_data(bmi_gwas)

head(bmi_exp_dat)
```


### 4. Outcome Data Example
(Optional) You can import outcome data using the IEU API, as well.
By taking snps argument, we can scope the common variant list with BMI  
```
chd_out_dat <- extract_outcome_data(
    snps = bmi_exp_dat$SNP,
    outcomes = 'ieu-a-7'
)
```
In this practice, we are going to import outcome data using GWAS summary file from [GWAS catalog](https://www.ebi.ac.uk/gwas/).
```
chd_outcome_dat <- read_outcome_data(
    snps = bmi_exp_dat$SNP,
    filename = "/data/GCDA/4_MR/26343387-GCST003116-EFO_0000378.h.tsv",
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
head(chd_out_dat)
```


### 5. Harmonizing Data
Harmonizing data step is required because there are mismatches between alleles in exposure and outcome. Recent GWASs typically present the effects of a SNP in reference to the allele on the forward strand but not necessarily.
![image](https://user-images.githubusercontent.com/22064612/235645818-0c25fa91-7ba8-4dcf-9066-406e2db0eb9e.png)

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
Let's try MR with Egger, two sample maximum likelihood,simple median, and IVW. You can add list of methods or methods set by default.

```
res <- mr(dat, method_list = c("mr_egger_regression", "mr_two_sample_ml","mr_simple_median","mr_ivw"))
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
Let's save to pdf using ggsave() from ggplot2
```
library(ggplot2)

p1 <- mr_scatter_plot(res, dat)
print(length(p1))
ggplot2::ggsave(p1[[1]], file = "/data/home/USER_ID/Scatter.png", width = 7, height = 7)
```
Download the plot to local computer.
```
scp USER_ID@147.47.200.131 USER_ID@leelabsg15:/data/home/USER_ID/Scatter.png ./
scp -P 22555 -J USER_ID@147.47.200.131 USER_ID@leelabsg15:/data/home/USER_ID/Scatter.png ./
```
![filename](https://user-images.githubusercontent.com/22064612/235652335-b57f0381-b820-47e0-a07f-da09564f9c0b.png)


Single SNP MR can be also performed by following. We can visualize the results via forest and funnel plots
```
res_single <- mr_singlesnp(dat)
head(res_single)
p2 <- mr_forest_plot(res_single)
ggplot2::ggsave(p2[[1]], file = "/data/home/USER_ID/Forest.png", width = 7, height = 7)
p3 <- mr_funnel_plot(res_single)
ggplot2::ggsave(p3[[1]], file = "/data/home/USER_ID/Funnel.png", width = 7, height = 7)
```
To see if a single snp is driving the association, we can perform leave-one-out MR.
In this case, we can see that there is no single variant that driving the association between exposure and outcome.
```
res_loo <- mr_leaveoneout(dat)
head(res_loo) ; summary(res_loo$p)
p4 <- mr_leaveoneout_plot(res_loo)
ggplot2::ggsave(p4[[1]], file = "/data/home/USER_ID/LOO.png", width = 7, height = 7)
```




