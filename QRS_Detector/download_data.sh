output_folder=/home/lucas/Projects/ADNE_Project/QRS_Detector/data

for i in $(seq 1 9)
do
	wget https://physionet.org/pn3/incartdb/I0$i.dat -P $output_folder
	wget https://physionet.org/pn3/incartdb/I0$i.hea -P $output_folder
	wget https://physionet.org/pn3/incartdb/I0$i.atr -P $output_folder
done

for j in $(seq 10 75)
do
	wget https://physionet.org/pn3/incartdb/I$j.dat -P $output_folder
        wget https://physionet.org/pn3/incartdb/I$j.hea -P $output_folder
        wget https://physionet.org/pn3/incartdb/I$j.atr -P $output_folder
done
