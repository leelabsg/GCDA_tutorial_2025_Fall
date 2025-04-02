# Practice Session #2: Genome-wide Association Studies (GWAS) (April 3, 2025)

In this session, we will learn how to conduct the genome-wide association analysis using SAIGE. \
(References: [Paper](https://www.nature.com/articles/s41588-018-0184-y), [Github](https://github.com/saigegit/SAIGE), [Documentation](https://saigegit.github.io/SAIGE-doc/)) \
This document was created on April 3, 2025 and the following contents were tested on the Lee Lab cluster (Ubuntu 24.04.1 LTS).

### 1. Setting up the environment

We will use [Docker](https://www.docker.com/) on the Lee Lab cluster (`leelabsg15` node). \
It is already created on the GSDS cluster, but you can create the environment on your local machine with the following command:

```
# Pull SAIGE docker image
sudo docker pull wzhou88/saige:1.4.4

# Give previlege to use docker without sudo
sudo usermod -aG docker $USER
```

You can test if it works:

```
docker run --rm -it -v /data/home/gcda_XXX/GCDA/2_GWAS:/data/home/gcda_XXX/GCDA/2_GWAS wzhou88/saige:1.4.4 /bin/bash
```

### 2. Preparing data

We need (individual-level) genotype and phenotype files to conduct GWAS. \
However, access to these files is strictly restricted.

Let's take a quick look at what the real data (UK Biobank) looks like.

#### Phenotype file

The phenotype file of UK Biobank looks like the following:

```
f.eid f.46.0.0 f.47.0.0
1000001 18 21
1000002 32 44
1000003 45 42
1000004 42 44
1000005 51 48
1000006 53 45
1000007 28 30
1000008 29 27
1000009 33 32
```

You can find the information of each phenotype column in [UK Biobank showcase](https://biobank.ndph.ox.ac.uk/showcase/).

* Field ID 46 means 'hand grip strength (left)'
* Field ID 47 means 'hand grip strength (right)'

#### Genotype file

##### Variant Call Format (`vcf`)

It contains meta-information lines, a header line, and then data lines each containing information about a position in the genome.


##### PLINK binary (`bed`, `bim`, `fam`)

PLINK binary files are the binary version of PLINK files (`ped`, `map`). \
PLINK binary genotype data consists of 3 files:

* `bed` file: Genotype data in binary format
* `bim` file: Variants information
* `fam` file: Sample information

`bim` file contains the information of variants.

```
1	rs1	0	1	C	A
1	rs3	0	3	C	A
1	rs4	0	4	C	A
1	rs6	0	6	C	A
1	rs7	0	7	C	A
1	rs8	0	8	C	A
1	rs9	0	9	C	A
1	rs10	0	10	C	A
1	rs11	0	11	C	A
1	rs12	0	12	C	A
```

`fam` file contains the information of samples.

```
1a1	1a1	0	0	0	-9
1a2	1a2	0	0	0	-9
1a3	1a3	0	0	0	-9
1a4	1a4	0	0	0	-9
1a5	1a5	0	0	0	-9
1a6	1a6	0	0	0	-9
1a7	1a7	0	0	0	-9
1a8	1a8	0	0	0	-9
1a9	1a9	0	0	0	-9
1a10	1a10	0	0	0	-9
```

`bed` file contains the genotype information (in binary format). \
We cannot easily check the genotype information, but we can see the bitwise 

```
00000000: 01101100 00011011 00000001 11111111 11111111 11111111  l.....
00000006: 11111111 11111111 11111111 11111111 11111111 11101111  ......
0000000c: 11111011 11111111 11111111 11111111 11111111 11111111  ......
00000012: 11111111 11111111 11111111 11111111 11111111 11111111  ......
00000018: 11111111 11111111 11111111 11111111 11111111 11111111  ......
0000001e: 11111111 11111111 11111111 11111111 11111111 10111111  ......
00000024: 11111110 11111011 11111111 11111111 10111111 11111011  ......
0000002a: 11111111 11111111 11111111 11111111 11111111 11111111  ......
00000030: 11111111 11111111 11111111 11111111 11111111 11111111  ......
00000036: 11111111 11111111 11111111 11111111 11111111 11111111  ......
```

You can find more details on [PLINK website](https://zzz.bwh.harvard.edu/plink/binary.shtml).


#### Example Data

We will use `pheno_example.txt` as a phenotype file.

```
IID	x1	                x2	y_binary  y_quantitative
1a1	1.51178116845085	1	0	  2.0046544617651
1a2	0.389843236411431	1	0	  0.104213400269085
1a3	-0.621240580541804	1	0	  -0.397498354133647
1a4	-2.2146998871775	1	0	  -0.333177899030597
1a5	1.12493091814311	1	0	  1.21333962248852
1a6	-0.0449336090152309	1	0	  -0.275411643032321
1a7	-0.0161902630989461	0	0	  0.438532936074923
1a8	0.943836210685299	0	0	  0.0162938047248591
1a9	0.821221195098089	1	0	  0.147167262428064
```


### 3. Run GWAS using SAIGE

#### What is SAIGE?

* Mixed effect model-based method
* Score test to compute test statistics
* Several techniques for fast computation
For detailed explanation, please refer the [Notion page](https://admitted-guan-7d5.notion.site/SAIGE-10ea25444f7a802ea917d4aad44f2ce0).

#### Why SAIGE?

* Can account for related individuals
* Fast computation
* Prevent type I error inflation in case of case-control imbalance

#### Process

##### Step 0. (Optional) Create sparse GRM

This sparse GRM only needs to be created once for each data set, e.g. a biobank, and can be used for all different phenotypes as long as all tested samples are in the sparse GRM.

```
docker run -v /data/home/gcda_XXX/GCDA/2_GWAS:/data/home/gcda_XXX/GCDA/2_GWAS \
        -v /data/GCDA/usr:/data/GCDA/usr \
        wzhou88/saige:1.4.4 createSparseGRM.R \
        --plinkFile=/data/home/gcda_XXX/GCDA/2_GWAS/genotype/nfam_100_nindep_0_step1_includeMoreRareVariants_poly_22chr \
        --nThreads=4 \
        --outputPrefix=/data/home/gcda_XXX/GCDA/2_GWAS/sparseGRM \
        --numRandomMarkerforSparseKin=2000 \
        --relatednessCutoff=0.125
```

##### Step 1. Fitting the null (logistic/linear) mixed model

###### Binary trait example

```
docker run -v /data/home/gcda_XXX/GCDA/2_GWAS:/data/home/gcda_XXX/GCDA/2_GWAS \
        wzhou88/saige:1.4.4 step1_fitNULLGLMM.R \
        --plinkFile=/data/home/gcda_XXX/GCDA/2_GWAS/genotype/nfam_100_nindep_0_step1_includeMoreRareVariants_poly_22chr \
        --phenoFile=/data/home/gcda_XXX/GCDA/2_GWAS/phenotype/pheno_example.txt \
        --phenoCol=y_binary \
        --covarColList=x1,x2 \
        --sampleIDColinphenoFile=IID \
        --traitType=binary \
        --outputPrefix=/data/home/gcda_XXX/GCDA/2_GWAS/example_binary \
        --nThreads=8 \
        --IsOverwriteVarianceRatioFile=TRUE
```

###### Quantitative trait example

```
docker run -v /data/home/gcda_XXX/GCDA/2_GWAS:/data/home/gcda_XXX/GCDA/2_GWAS \
        wzhou88/saige:1.4.4 step1_fitNULLGLMM.R \
        --plinkFile=/data/home/gcda_XXX/GCDA/2_GWAS/genotype/nfam_100_nindep_0_step1_includeMoreRareVariants_poly_22chr \
        --phenoFile=/data/home/gcda_XXX/GCDA/2_GWAS/phenotype/pheno_example.txt \
        --phenoCol=y_quantitative \
        --invNormalize=TRUE \
        --covarColList=x1,x2 \
        --sampleIDColinphenoFile=IID \
        --traitType=quantitative \
        --outputPrefix=/data/home/gcda_XXX/GCDA/2_GWAS/example_quantitative \
        --nThreads=8 \
        --IsOverwriteVarianceRatioFile=TRUE
```

You can find more example usages in [SAIGE Documentation page](https://saigegit.github.io/SAIGE-doc/docs/single_example.html).

##### Step 2. Performing single-variant association tests

###### Quantitative trait example

```
docker run -v /data/home/gcda_XXX/GCDA/2_GWAS:/data/home/gcda_XXX/GCDA/2_GWAS \
        wzhou88/saige:1.4.4 step2_SPAtests.R \
        --vcfFile=/data/home/gcda_XXX/GCDA/2_GWAS/genotype/genotype_100markers.vcf.gz \
        --vcfFileIndex=/data/home/gcda_XXX/GCDA/2_GWAS/genotype/genotype_100markers.vcf.gz.csi \
        --vcfField=GT \
        --chrom=1 \
        --minMAF=0 \
        --minMAC=10 \
        --GMMATmodelFile=/data/home/gcda_XXX/GCDA/2_GWAS/example_quantitative.rda \
        --varianceRatioFile=/data/home/gcda_XXX/GCDA/2_GWAS/example_quantitative.varianceRatio.txt \
        --SAIGEOutputFile=/data/home/gcda_XXX/GCDA/2_GWAS/step2_quant.txt
```


### 4. Drawing plots

Using `qqman` package in R, we can draw manhattan plot and Q-Q plot with the GWAS result (summary statistics). ([Documentation](https://cran.r-project.org/web/packages/qqman/qqman.pdf))

You can copy files to your local machine (from the server) using `scp` command.

```
scp 'gcda_XXX@147.47.200.131:SOURCE_PATH' DESTINATION_PATH

# Example
scp 'gcda_XXX@147.47.200.131:~/GCDA/usr/YOUR_DIRECTORY/*.png' .
```

And RStudio server is also available on the Lee Lab cluster: http://147.47.200.131:8786/

```
# Draw manhattan and Q-Q plot (RStudio server)
library(qqman)

gwas <- read.table("~/GCDA/2_GWAS/step2_quant.txt", header=T)

manhattan(gwas, main='Manhattan Plot', chr = "CHR", bp = "POS", snp = "MarkerID", p = "p.value")

qq(gwas$p.value, main='QQ plot')
```

Or you can use [LocusZoom](http://locuszoom.org/) or [PheWeb](https://github.com/statgen/pheweb) to visualize your GWAS results and host them on the web server.
