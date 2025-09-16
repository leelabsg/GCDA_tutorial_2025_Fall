# Practice Session #1: Sequencing (Sep. 16, 2025)

In this session, we will learn how to convert raw unmapped read files (`FASTQ`) to analysis-ready files (`VCF`). \
The overall process in this session is mainly based on the [GATK Best Practice](https://gatk.broadinstitute.org/hc/en-us/categories/360002302312-Getting-Started). \
This document was created on Mar. 19, 2025 and the following contents were tested on local WSL (Ubuntu 22.04.1 LTS) + GSDS Cluster.
### 0. Installing Linux and Mamba(Anaconda) in Windows
Using Linux has become easy in Windows with WSL. \
To start, launch windows powershell in administration mode and run following. 
``` 
wsl --install
```
After system restart, linux can be run from terminal app. (Note that hard drive is mounted under /mnt)

find latest release for Linux-x86 and copy link from https://www.anaconda.com/products/distribution

install anaconda by following instructions

### Access Leelab computational server via personal account
To access to leelab computational server, you can use ssh command from any command shell, or vscode
Your ID is gcda_<xxx>, last three digits of student ID
```
ssh gcda_<xxx>@147.47.200.131 -p 22555
ID: gcda_<xxx>
PW (default): gcda_<xxx>

```
#### Installing mamba for faster conda usage
Follow the instruction from https://github.com/conda-forge/miniforge 

```
bash Miniforge3-Linux-x86_64.sh -b -p "${HOME}/conda"
conda/bin/conda init
```

### Use designated node for computation: leelabsg11
After login, use ssh to log on to designated node for GCDA 2025 Fall
```
ssh leelabsg11
```

### 1. Setting up the environment

We will use the Anaconda environment on the GSDS cluster. \
It is already created on the GSDS cluster, but you can create the environment on your local machine with the following command \
In this session, OpenJDK, samtools, GATK and BWA are installed in creation of conda environment and Picard is downloaded as java package \
All files and tools are included in '~/1_sequencing/' folder
```
# Create conda/mamba environment and install softwares 
# conda create -n SEQ samtools bwa -c anaconda -c bioconda
mamba create -n SEQ samtools bwa -c anaconda -c bioconda

conda activate SEQ

# Install jdk 17 version (Required after picard 3.0.0)
## wget https://download.java.net/java/GA/jdk17.0.2/dfd4a8d0985749f896bed50d7138ee7f/8/GPL/openjdk-17.0.2_linux-x64_bin.tar.gz
tar xvf ~/1_sequencing/utils/openjdk-17.0.2_linux-x64_bin.tar.gz
export JAVA_HOME=$JAVA_HOME/:~/jdk-17.0.2/
export PATH=$JAVA_HOME/bin:$PATH
alias java17="~/jdk-17.0.2/bin/java"

# install gatk4 
conda install gatk4 -c bioconda
mamba install gatk4 -c bioconda
# Download Picard (Find Latest Release: https://github.com/broadinstitute/picard/releases/latest)
## cd ~/1_sequencing/utils
## wget https://github.com/broadinstitute/picard/releases/download/3.0.0/picard.jar

```

### 2. Preparing data

To map our raw unmapped reads, we need the reference panel and the information for known variants. \
Here, we will use the `FASTA` file of 1000 Genome Phase 3 (GRCh37 build) and the `VCF` file for known variants. \
You can browse [FTP server](ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/) of 1000 Genome Project.

#### Already done in your user account
```
mkdir -p ~/1_sequencing/reference
cd ~/1_sequencing/reference
# Download 1000 Genome reference panel
## wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/human_g1k_v37.fasta.gz
## gzip -d human_g1k_v37.fasta.gz

# Download VCF file and its index (tbi) file for known variants
## wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/integrated_sv_map/ALL.wgs.mergedSV.v8.20130502.svs.genotypes.vcf.gz
## wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/integrated_sv_map/ALL.wgs.mergedSV.v8.20130502.svs.genotypes.vcf.gz.tbi
```

You can check the contents of the `FASTA` file by the following command:

```
# View first 200 rows of FASTA file
head -200 human_g1k_v37.fasta
```

We need to preprocess (create index/dict files) the reference genome files.

```
# Create index (.fai)
samtools faidx human_g1k_v37.fasta

# Create dict (.dict)
gatk CreateSequenceDictionary -R human_g1k_v37.fasta

# Construct files with Burrows-Wheeler Transformation (5 files)
## bwa index -a bwtsw human_g1k_v37.fasta
```

And we need a sequence read file (`FASTQ`) for the sample individual (HG00096).
This can be downloaded from ftp server wih project description : [1000 Genome Project Phase 3](https://www.internationalgenome.org/1000-genomes-summary)
```
cd ~/1_sequencing/raw_reads
# Download sequence read file from 1000 Genome
# sample HG00096
## wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00096/sequence_read/SRR062634.filt.fastq.gz
## gzip -d SRR062634.filt.fastq.gz
# sample HG00097
## wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00097/sequence_read/SRR741384.filt.fastq.gz
## gzip -d SRR741384.filt.fastq.gz
# sample HG00099
## wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00099/sequence_read/SRR741411.filt.fastq.gz
## gzip -d SRR741411.filt.fastq.gz
```

We can check the contents of the `FASTQ` file by `head` command, and you will see the following:

```
@SRR062634.321 HWI-EAS110_103327062:6:1:1446:951/2
TGATCATTTGATTAATACTGACATGTAGACAAGAAGAAAAGTATGTTTCATGCTATTTTGAGTAACTTCCATTTAGAAGCCTACTCCTGAGCACAACATT
+
B5=BD5DAD?:CBDD-DDDDDCDDB+-B:;?A?CCE?;D3A?B?DB??;DDDEEABD+>DAC?A-CD-=D?C5A@::AC-?AB?=:>CA@##########
@SRR062634.488 HWI-EAS110_103327062:6:1:1503:935/2
AATGTTATTAAAAATGGACACCTTTTTCTCACACATTCAGTTTCATTGTCTCGCACCCCATCGTTTTACTTTTCTTCCTTCAGAAAATGATAAATGTGGG
+
AAAA?5D?BD==ADBD:DBDDDDD5D=;@>AD-CD?D=C5=@4<7CCAA5?=?>5@BC?*<:=>>:D:B5?B?5?'3::5?5<:;*97:<A#########
@SRR062634.849 HWI-EAS110_103327062:6:1:1587:921/2
CAGATCAGAATAATTTTTGTGTTATGTACGTGTAAGAAAACATAGCTATTATGATATGGAAACTAGGAGTGAAATATGAGGAATTTGTGACTTTTCTGAA
```

A `FASTQ` file normally uses four lines per sequence.

* **Line 1** begins with a '@' character and is followed by a **sequence identifier** and an optional description (like a `FASTA` title line).
* **Line 2** is the **raw sequence letters**.
* **Line 3** begins with a '+' character and is **optionally followed by the same sequence identifier** (and any description) again.
* **Line 4** encodes the **quality values** for the sequence in Line 2, and must contain the same number of symbols as letters in the sequence.

You can find the information of `SRR062634` in the [Sequence Read Archive (SRA) page](https://www.ncbi.nlm.nih.gov/sra/?term=srr062634) of NCBI website.

Here are the quality value characters in left-to-right increasing order of quality (ASCII):

```
!"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~
```

You can learn more about `FASTQ` files on [Wikipedia](https://en.wikipedia.org/wiki/FASTQ_format).

Many of individuals in 1000 Genome Project have multiple `FASTQ` files, because many of them were sequenced using more than one run of a sequencing machine. \
Each set of files named like `SRR062634_1.filt.fastq.gz`, `SRR062634_2.filt.fastq.gz` and `SRR062634.filt.fastq.gz` represent all the sequence from a sequencing run.

The labels with `_1` and `_2` represent paired-end files, and the files which do not have a number in their name are single-ended reads. (or if one of a pair of reads gets rejected the other read gets placed in the single file.)

### 3. Overall Process

This practice session consists of 4 steps.

1) Preprocess the `FASTQ` file
 * `FastqToSam`
 * `AddOrReplaceReadGroups`
 * `MarkIlluminaAdapters`
 * `SamToFastq`
2) Convert `FASTQ` to `BAM`
 * Map `FASTQ` file to reference with `bwa mem`
 * Merge bam with index alignment (`MergeBamAlignment`)
 * Mark duplicated reads ([`MarkDuplicates`](https://gatk.broadinstitute.org/hc/en-us/articles/360037052812-MarkDuplicates-Picard-))
 * Sort bamfile (`SortSam`)
 * [Base Quality Score Recalibration](https://gatk.broadinstitute.org/hc/en-us/articles/360035890531-Base-Quality-Score-Recalibration-BQSR-) (`BaseRecalibrator`)
 * `ApplyBQSR`
3) Convert `BAM` to `GVCF`
 * `HaplotypeCaller`
4) Convert `GVCF` to `VCF`
 * `GenotypeGVCFs`

### 4. Preprocessing the `FASTQ` file

#### Convert the raw `FASTQ` file to an unmapped `BAM` file

Using `FastqToSam` function of Picard, we can convert the `FASTQ` file to an unmapped `BAM` file.

```
mkdir -p ~/1_sequencing/data/
SID=HG00096
java17 -jar ~/1_sequencing/utils/picard.jar FastqToSam \
F1=~/1_sequencing/raw_reads/SRR062634.filt.fastq \
O=~/1_sequencing/data/fastq_to_bam_${SID}.bam \
SM=${SID}
```

#### Add read group information in `BAM` file

We can add read groups in `BAM` file by the following command:

```
cd ~/1_sequencing/data/
java17 -jar ~/1_sequencing/utils/picard.jar AddOrReplaceReadGroups \
I=fastq_to_bam_${SID}.bam \
O=add_read_groups_${SID}.bam \
RGID=4 \
RGLB=lib1 \
RGPL=ILLUMINA \
RGPU=unit1 \
RGSM=20
```

#### Mark adapter sequences

Using `MarkIlluminaAdapters`, we can mark adapter sequences.

```
java17 -Xmx8G -jar ~/1_sequencing/utils/picard.jar MarkIlluminaAdapters \
I=add_read_groups_${SID}.bam \
O=mark_adapter_${SID}.bam \
M=mark_adapter_${SID}.metrics.txt
```

#### Convert the preprocessed `BAM` file to a `FASTQ` file

```
java17 -Xmx8G -jar ~/1_sequencing/utils/picard.jar SamToFastq \
I=mark_adapter_${SID}.bam \
FASTQ=fastq_input_${SID}.fq \
CLIPPING_ATTRIBUTE=XT \
CLIPPING_ACTION=2 \
INTERLEAVE=true \
NON_PF=true
```

### 5. Convert the preprocessed `FASTQ` file to an aligned `BAM` file

#### Align reads using `BWA MEM`

GATK's variant discovery workflow recommends Burrows-Wheeler Aligner's maximal exact matches (BWA-MEM) algorithm.

```
bwa mem -M -t 7 -p ~/1_sequencing/reference/human_g1k_v37.fasta ~/1_sequencing/data/fastq_input_${SID}.fq > aligned_${SID}.sam
```

You can check the contents of the `SAM` file:

```
@SQ	SN:1	LN:249250621
@SQ	SN:2	LN:243199373
@SQ	SN:3	LN:198022430
...
SRR062634.10000020	16	1	246002445	60	100M	*	0	0	ACAGCACCAGGCCAGCCTTTTTATTTTATTTTAATTTTTATTATTTTGAGACATTCTCGCTCTTTCGCCCAGGCCGGACTGCAGTGGTGCTATCTCAGCT	<C==C:@?A=9:5,=?>=?C=?;EE<CCBCAC=/?@FFAEEEEBEDEEABCFCBBGGEBFEDFCEGFGDF?GAGGGGFGGGGGGGGGGGGFGGGFGGGGG	NM:i:0	MD:Z:100	AS:i:100	XS:i:32
SRR062634.10000713	16	2	39841450	60	100M	*	0	0	TTAGCCATTCTAGTAGCTGTGTAGCAATTATGCTAGTTAACTGGTCAAATCTAATAGAGATGCTATCTAAAATGTGTTATAAAGAATGTGACTTGAGAGT	==:=C@A??DC@CC@CEBCCCCEDEGD=EECEF?EEBEGEFEFEEEGFGEGEGGDGFGFGEDGFGGGGGGGGEGGGGGGGGGGGGGGGGGGGGGGGGGGG	NM:i:0	MD:Z:100	AS:i:100	XS:i:19
SRR062634.10000906	0	5	97444533	60	100M	*	0	0	CAGTTTGATCCTTCTGAATTAGATTTTCCATACATGAAGCCTATGGGACTCTGGTGGGCAGTAGAAGATAAACTGTAATTTAAGTGAGGTTTTTATAAGC	EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE5BDDCC?EEEEEEDBEEEEEEEDEEEEA?DEEEEBE@EEEEADEDEE@AD?E	NM:i:1	MD:Z:48T51	AS:i:95	XS:i:20
```

The header section must be prior to the alignment section if it is present. Headings begin with the '@' symbol, which distinguishes them from the alignment section. \
Alignment sections have 11 mandatory fields, as well as a variable number of optional fields. \
SAM format explained: https://samtools.github.io/hts-specs/SAMv1.pdf \
SAM, BAM, and CRAM https://gatk.broadinstitute.org/hc/en-us/articles/360035890791-SAM-or-BAM-or-CRAM-Mapped-sequence-data-formats

The information of some columns are as follows:

* Column 1: Query Template Name (QNAME)
* Column 2: Bitwise Flag 
* Column 3: Reference sequence name (RNAME), often contains the Chromosome name
* Column 4: Leftmost position of where this alignment maps to the reference (POS)
* Column 5: Mapping Quality
* Column 6: Concise Idiosyncratic Gapped Alignment Report (CIGAR) string ([Wikipedia](https://en.wikipedia.org/wiki/Sequence_alignment#Representations))
* Column 10: Segment sequence
* Column 11: ASCII representation of phred-scale base quality

#### Add information to `BAM` file using `MergeBamAlignment`

```
java17 -Xmx10G -jar ~/1_sequencing/utils/picard.jar MergeBamAlignment \
R=~/1_sequencing/reference/human_g1k_v37.fasta \
UNMAPPED=add_read_groups_${SID}.bam \
ALIGNED=aligned_${SID}.sam \
O=preprocessed_${SID}.bam \
CREATE_INDEX=true \
ADD_MATE_CIGAR=true \
CLIP_ADAPTERS=false \
CLIP_OVERLAPPING_READS=true \
INCLUDE_SECONDARY_ALIGNMENTS=true \
MAX_INSERTIONS_OR_DELETIONS=-1  \
PRIMARY_ALIGNMENT_STRATEGY=MostDistant \
ATTRIBUTES_TO_RETAIN=XS
```

#### Mark Duplicates

```
java17 -jar ~/1_sequencing/utils/picard.jar MarkDuplicates \
I=preprocessed_${SID}.bam \
O=mark_dup_${SID}.bam \
M=mark_dup_${SID}.metrics.txt
```

#### Sort, index and convert alignment to a BAM using SortSam

```
java17 -jar ~/1_sequencing/utils/picard.jar SortSam \
I=mark_dup_${SID}.bam \
O=sorted_${SID}.bam \
SO=coordinate 
```

#### Create Recalibration Table using `BaseRecalibrator`

```
gatk --java-options '-Xmx10g' BaseRecalibrator \
-I sorted_${SID}.bam \
-R ~/1_sequencing/reference/human_g1k_v37.fasta \
--known-sites ~/1_sequencing/reference/ALL.wgs.mergedSV.v8.20130502.svs.genotypes.vcf.gz \
-O recal_data_${SID}.table
```

#### Base Quality Score Recalibration (BQSR)

```
gatk --java-options '-Xmx10g' ApplyBQSR \
-I sorted_${SID}.bam \
-R ~/1_sequencing/reference/human_g1k_v37.fasta \
--bqsr-recal-file recal_data_${SID}.table \
-O bqsr_${SID}.bam
```
#### Convert other samples into processed bam
```
for SID in HG00096 HG00097 HG00098
do
bash ~/1_sequencing/bam_proc.sh ${SID}
done
```


#### IGV Viewer (Software for visualization of `BAM` file)

We can visualize the aligned `BAM` file with the [IGV viewer](https://software.broadinstitute.org/software/igv/home).

For example, we can observe high coverage around SUMO1P1 gene. (`HG00096.chrom20.ILLUMINA.bwa.GBR.exome.20120522.bam`)
```
cd ~/1_sequencing/raw_reads
## wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00096/exome_alignment/HG00096.chrom20.ILLUMINA.bwa.GBR.exome.20120522.bam
## wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00096/exome_alignment/HG00096.chrom20.ILLUMINA.bwa.GBR.exome.20120522.bam.bai
## wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00097/exome_alignment/HG00097.chrom20.ILLUMINA.bwa.GBR.exome.20130415.bam
## wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00097/exome_alignment/HG00097.chrom20.ILLUMINA.bwa.GBR.exome.20130415.bam.bai
## wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00099/exome_alignment/HG00099.chrom20.ILLUMINA.bwa.GBR.exome.20130415.bam
## wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00099/exome_alignment/HG00099.chrom20.ILLUMINA.bwa.GBR.exome.20130415.bam.bai
```

### 6. Converting `BAM` to `GVCF`

#### Convert the individual `BAM` files to `GVCF` files

In this section, we will use the real data of 3 individuals in 1000 Genome Project (`HG00096`, `HG00097`, `HG00099`). \
We can convert these `BAM` files to `GVCF` files.

```
cd ~/1_sequencing/data
# Convert BAM for the first individual
gatk --java-options "-Xms4g" HaplotypeCaller \
-R ~/1_sequencing/reference/human_g1k_v37.fasta \
-I ~/1_sequencing/raw_reads/HG00096.chrom20.ILLUMINA.bwa.GBR.exome.20120522.bam \
-L 20 \
-ERC GVCF \
-O sample01_20.g.vcf
```

```
# Convert BAM for the second individual
gatk --java-options "-Xms4g" HaplotypeCaller \
-R ~/1_sequencing/reference/human_g1k_v37.fasta \
-I ~/1_sequencing/raw_reads/HG00097.chrom20.ILLUMINA.bwa.GBR.exome.20130415.bam \
-L 20 \
-ERC GVCF \
-O sample02_20.g.vcf
```

```
# Convert BAM for the third individual
gatk --java-options "-Xms4g" HaplotypeCaller \
-R ~/1_sequencing/reference/human_g1k_v37.fasta \
-I ~/1_sequencing/raw_reads/HG00099.chrom20.ILLUMINA.bwa.GBR.exome.20130415.bam \
-L 20 \
-ERC GVCF \
-O sample03_20.g.vcf
```
#### What GVCF provides us : SNV
```
bcftools view -v snps sample03_20.g.vcf | grep -v '##' | head
```
#### What GVCF provides us : Insertion/Deletion
```
bcftools view -v indels sample03_20.g.vcf | grep -v '##' | head
```
<img width="850" height="578" alt="image" src="https://github.com/user-attachments/assets/89a3c5c8-39e2-44fa-af8f-ac40344431ab" />

#### Combine individual `GVCF` files

```
gatk CombineGVCFs \
-R ~/1_sequencing/reference/human_g1k_v37.fasta \
--variant sample01_20.g.vcf \
--variant sample02_20.g.vcf \
--variant sample03_20.g.vcf \
-O sample_all.g.vcf.gz
```

### 7. Converting `GVCF` to `VCF`

```
gatk --java-options "-Xmx4g" GenotypeGVCFs \
-R ~/1_sequencing/reference/human_g1k_v37.fasta \
-V sample_all.g.vcf.gz \
-O sample_all.vcf
```

You can check the contents of the final `VCF` file.

```
##fileformat=VCFv4.2
##ALT=<ID=NON_REF,Description="Represents any possible alternative allele not already represented at this location by REF and ALT">
##FILTER=<ID=LowQual,Description="Low quality">
...
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG00096	HG00097	HG00099
20	61795	.	G	T	39.24	.	AC=2;AF=1.00;AN=2;DP=2;ExcessHet=0.0000;FS=0.000;MLEAC=2;MLEAF=1.00;MQ=60.00;QD=19.62;SOR=0.693	GT:AD:DP:GQ:PL	1/1:0,2:2:6:49,6,0	./.:0,0:0:0:0,0,0	./.:0,0:0:0:0,0,0
20	68749	.	T	C	193.81	.	AC=2;AF=0.500;AN=4;DP=11;ExcessHet=0.0000;FS=0.000;MLEAC=2;MLEAF=0.500;MQ=60.00;QD=25.36;SOR=3.611	GT:AD:DP:GQ:PL	1/1:0,5:5:15:208,15,0	./.:0,0:0:0:0,0,0	0/0:4,0:4:12:0,12,115
20	76962	.	T	C	19086.73	.	AC=6;AF=1.00;AN=6;BaseQRankSum=1.80;DP=539;ExcessHet=0.0000;FS=0.000;MLEAC=6;MLEAF=1.00;MQ=59.41;MQRankSum=-3.240e-01;QD=28.73;ReadPosRankSum=1.42;SOR=0.260	GT:AD:DP:GQ:PL	1/1:0,357:357:99:14330,1073,0	1/1:1,65:66:99:2125,188,0	1/1:0,84:84:99:2645,250,0
```

We use this `VCF` file in the analysis!

### 8. Variant Calling with Deepvariant (Optional) 
DeepVariant is a deep learning-based variant caller that takes aligned reads (in BAM or CRAM format), produces pileup image tensors from them, classifies each tensor using a convolutional neural network, and finally reports the results in a standard VCF or gVCF file. (https://github.com/google/deepvariant)

The simple concept of Deepvariant is explained and visualized in following post:
[Looking Through DeepVariant's Eyes](https://google.github.io/deepvariant/posts/2020-02-20-looking-through-deepvariants-eyes/)
