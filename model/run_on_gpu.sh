# ssh -p [PORT] -i ~/.ssh/mli_computa root@[IP]

apt-get update
apt-get install git
apt-get install git-lfs
git lfs install
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

cd ~
git clone https://github.com/dhedey/mlx-8-backprop-week-1-hacknews-vote-predictions.git
cd mlx-8-backprop-week-1-hacknews-vote-predictions

uv run ./model/train.py --batch-size 1024