# Automated downloading of required data files

data_dir = "./data/raw"

cmp_base = "https://cog.sanger.ac.uk/cmp/download"
dgidb_base = "https://www.dgidb.org/data/monthly_tsvs/2022-Feb"
gdsc_base = "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.4"
gdsc_api_base = "https://www.cancerrxgene.org/api"
stringdb_base = "https://stringdb-downloads.org/download"

all: cmp gdsc depmap dgidb

cmp:
	mkdir -p $(data_dir)/CellModelPassports
	echo "Downloading Cell Model Passports data..."
	wget --no-check-certificate -P $(data_dir)/CellModelPassports $(cmp_base)/gene_identifiers_20191101.csv
	wget --no-check-certificate -P $(data_dir)/CellModelPassports $(cmp_base)/mutations_all_20230202.zip
	wget --no-check-certificate -P $(data_dir)/CellModelPassports $(cmp_base)/rnaseq_all_20220624.zip
	wget --no-check-certificate -P $(data_dir)/CellModelPassports $(cmp_base)/mutations_wes_vcf_20221010.zip
	wget --no-check-certificate -P $(data_dir)/CellModelPassports $(cmp_base)/WES_pureCN_CNV_genes_20221213.zip
	echo "Finished downloading Cell Model Passports data!"
	if [ -f "$(data_dir)/CellModelPassports/rnaseq_all_20220624.zip" ]; then \
		echo "Unpacking Cell Model Passports expression data..."; \
		unzip -d $(data_dir)/CellModelPassports $(data_dir)/CellModelPassports/rnaseq_all_20220624.zip; \
	fi
	if [ -f "$(data_dir)/CellModelPassports/mutations_all_20230202.zip" ]; then \
		echo "Unpacking Cell Model Passports mutation data..."; \
		unzip -d $(data_dir)/CellModelPassports $(data_dir)/CellModelPassports/mutations_all_20230202.zip; \
	fi
	if [ -f "$(data_dir)/CellModelPassports/mutations_wes_vcf_20221010.zip" ]; then \
		echo "Unpacking Cell Model Passports VCFs..."; \
		unzip -d $(data_dir)/CellModelPassports $(data_dir)/CellModelPassports/mutations_wes_vcf_20221010.zip; \
	fi
	if [ -f "$(data_dir)/CellModelPassports/WES_pureCN_CNV_genes_20221213.zip" ]; then \
		echo "Unpacking Cell Model Passports WES copy number data..."; \
		unzip -d $(data_dir)/CellModelPassports $(data_dir)/CellModelPassports/WES_pureCN_CNV_genes_20221213.zip; \
	fi

dgidb:
	mkdir -p $(data_dir)/DGIdb
	echo "Downloading DGIdb data..."
	wget --no-check-certificate -P $(data_dir)/DGIdb $(dgidb_base)/interactions.tsv
	wget --no-check-certificate -P $(data_dir)/DGIdb $(dgidb_base)/drugs.tsv
	wget --no-check-certificate -P $(data_dir)/DGIdb $(dgidb_base)/genes.tsv
	wget --no-check-certificate -P $(data_dir)/DGIdb $(dgidb_base)/categories.tsv
	echo "Finished downloading DGIdb data!"

gdsc:
	mkdir -p $(data_dir)/GDSCv1
	echo "Downloading GDSCv1 data..."
	wget --no-check-certificate -P $(data_dir)/GDSC $(gdsc_base)/GDSC1_fitted_dose_response_24Jul22.xlsx
	wget --no-check-certificate -P $(data_dir)/GDSC $(gdsc_base)/GDSC2_fitted_dose_response_24Jul22.xlsx
	wget --no-check-certificate -P $(data_dir)/GDSC $(gdsc_base)/Cell_Lines_Details.xlsx
	wget --no-check-certificate -P $(data_dir)/GDSC $(gdsc_base)/screened_compounds_rel_8.4.csv
	echo "Finished downloading GDSCv2 data!"

depmap:
	mkdir -p $(data_dir)/DepMap
	echo "Downloading DepMap data..."

stringdb:
	mkdir -p $(data_dir)/StringDB
	wget --no-check-certificate -P $(data_dir)/StringDB $(stringdb_base)/protein.links.v12.0/9606.protein.links.v12.0.txt.gz
	wget --no-check-certificate -P $(data_dir)/StringDB $(stringdb_base)/protein.physical.links.v12.0/9606.protein.physical.links.v12.0.txt.gz
	wget --no-check-certificate -P $(data_dir)/StringDB $(stringdb_base)/protein.info.v12.0/9606.protein.info.v12.0.txt.gz

.PHONY: all cmp gdsc gdsc depmap dgidb