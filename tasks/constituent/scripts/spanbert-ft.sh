#! /bin/sh

echo $(hostname)
echo $1
echo $2

# setup witter
cd /scratch/
mkdir freda
cd /scratch/freda
git clone https://github.com/ExplorerFreda/Witter.git
sleep $CUDA_VISIBLE_DEVICES
cd /scratch/freda/Witter
git checkout -- *
cd /scratch/freda/Witter/witter/slurm
git pull
bash install_conda.sh 
python /scratch/freda/Witter/witter/slurm/torch_version.py

# run probing script 
# 1: size; 2: method
export PYTORCH_PRETRAINED_BERT_CACHE=/share/data/lang/users/freda/.pytorch_pretrained_bert
export PATH=$PATH:/share/data/lang/users/freda/codebase/hackathon_2019/encoders/pretrained_transformers/SpanBERT/ 
export PYTHONPATH=$PYTHONPATH:/share/data/lang/users/freda/codebase/hackathon_2019/encoders/pretrained_transformers/SpanBERT/ 
cd /share/data/lang/users/freda/codebase/hackathon_2019
python -m tasks.constituent.main --model-type spanbert --model-name spanbert-$1-cased-$2-ft --model-size $1 \
    --encoding-method $2 --use-proj --proj-dim 256 --epochs 10 --fine-tune
