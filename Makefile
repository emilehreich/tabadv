DATA_PATH = data
TEXAS_PATH = ${DATA_PATH}/texas
KDDCUP99_PATH = ${DATA_PATH}/kddcup99
IEEECIS_PATH = ${DATA_PATH}/ieeecis

${TEXAS_PATH}/PUDF_base1_%q2013_tab.zip:
	@echo "By downloading the dataset you agree with its usage conditions: https://www.dshs.texas.gov/THCIC/Hospitals/Download.shtm"
	curl -G "https://www.dshs.texas.gov/thcic/hospitals/Data/PUDF_base1_$*q2013_tab/" > $@

${TEXAS_PATH}/PUDF_base1_%q2013_tab.txt: ${TEXAS_PATH}/PUDF_base1_%q2013_tab.zip
	@unzip $< -d ${TEXAS_PATH}

texas: ${TEXAS_PATH}/PUDF_base1_1q2013_tab.txt \
	   ${TEXAS_PATH}/PUDF_base1_2q2013_tab.txt \
	   ${TEXAS_PATH}/PUDF_base1_3q2013_tab.txt \
	   ${TEXAS_PATH}/PUDF_base1_4q2013_tab.txt

${KDDCUP99_PATH}/kddcup99.csv:
	curl -G "https://pkgstore.datahub.io/machine-learning/kddcup99/kddcup99_csv/data/5e61fb9132c9057c96cfb2b65aca7a93/kddcup99_csv.csv" > $@

kddcup99: ${KDDCUP99_PATH}/kddcup99.csv

${IEEECIS_PATH}/ieee-fraud-detection.zip:
	curl -G "https://drive.google.com/file/d/1im1DD421ypheVbQz4O1ANi5NccGk-G-t/view?usp=sharing" > $@

ieeecis: ${IEEECIS_PATH}/ieee-fraud-detection.zip
	unzip $< -d ${IEEECIS_PATH}

data: texas kddcup99 ieeecis

clean:
	rm -rf ${TEXAS_PATH}/*.zip
	rm -rf ${IEEECIS_PATH}/*.zip

.PHONY: data clean
