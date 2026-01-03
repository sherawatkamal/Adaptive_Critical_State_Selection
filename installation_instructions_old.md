# MAKE SURE TO RUN ALL OF THESE IN THE WEBSHOP DIRECTORY

conda env create -p /home/<YOUR USERNAME HERE>/env/acss --file acss_conda_environment.yml
source activate /home/<YOUR USERNAME HERE>/env/webshop  (this only works for arc because it requires source activate instead of conda activate)
pip install setuptools==65.5.0 
pip install wheel==0.38.0
pip install env train selenium thefuzz gym==0.24.0 transformers==4.57.1
pip install optimum-quanto accelerate transformers
pip install spacy
python -m spacy download en_core_web_sm
pip install en_core_web_sm
pip install wandb
pip install cleantext
pip install pyserini
pip install rank_bm25
pip install gdown
pip install evaluate


(also only for ARC)
module load Java
module load Miniconda3


cd data

gdown https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib
gdown https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu

gdown https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB
gdown https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi

gdown https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O 
cd ..

cd search_engine
mkdir -p resources resources_100 resources_1k resources_100k
python convert_product_file_format.py # convert items.json => required doc format
mkdir -p indexes
./run_indexing.sh


RUN THE FOLLOWING IN A BASH SCRIPT:

get_human_trajs () {
  PYCMD=$(cat <<EOF
import gdown
url="https://drive.google.com/drive/u/1/folders/16H7LZe2otq4qGnKw_Ic1dkt-o3U9Zsto"
gdown.download_folder(url, quiet=True, remaining_ok=True)
EOF
  )
  python -c "$PYCMD"
}
mkdir -p user_session_logs/
cd user_session_logs/
echo "Downloading 50 example human trajectories..."
get_human_trajs
echo "Downloading example trajectories complete"
cd ..



NOTE: feat_conv.pt and feat_ids.pt should be in /data folder