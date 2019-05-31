output_folder=/home/lucas/Projects/ADNE_Project/QRS_Detector/data/Test/

for j in $(seq 100 234)
do
	wget https://physionet.org/physiobank/database/mitdb/$j.dat -P $output_folder
        wget https://physionet.org/physiobank/database/mitdb/$j.hea -P $output_folder
        wget https://physionet.org/physiobank/database/mitdb/$j.atr -P $output_folder
done

