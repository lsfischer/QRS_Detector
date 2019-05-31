output_folder=/home/lucas/Projects/ADNE_Project/QRS_Detector/data/Test

for i in $(seq 1 9)
do
	wget https://physionet.org/pn3/incartdb/10$i.dat -P $output_folder
	wget https://physionet.org/pn3/incartdb/10$i.hea -P $output_folder
	wget https://physionet.org/pn3/incartdb/10$i.atr -P $output_folder
done

for j in $(seq 10 99)
do
	wget https://physionet.org/pn3/incartdb/1$j.dat -P $output_folder
        wget https://physionet.org/pn3/incartdb/1$j.hea -P $output_folder
        wget https://physionet.org/pn3/incartdb/1$j.atr -P $output_folder
done

for j in $(seq 100 234)
do
	wget https://physionet.org/pn3/incartdb/$j.dat -P $output_folder
        wget https://physionet.org/pn3/incartdb/$j.hea -P $output_folder
        wget https://physionet.org/pn3/incartdb/$j.atr -P $output_folder
done
