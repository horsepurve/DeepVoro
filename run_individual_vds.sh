mkdir dgr
mkdir dgr/5way5shot/
mkdir dgr/5way5shot/CUB
mkdir dgr/5way1shot/
mkdir dgr/5way1shot/CUB

sed -i '0,/XXX/{s/XXX/novel/}' run_ens_checkI.sh
sed -i '0,/XXX/{s/XXX/dgr/}' pipeline/run.py
bash run_ens_checkII.sh CUB 5 # DeepVoro 5-way 5-shot novel set
bash run_ens_checkII.sh CUB 1 # DeepVoro 5-way 1-shot novel set
sed -i '0,/novel/{s/novel/XXX/}' run_ens_checkI.sh
sed -i '0,/dgr/{s/dgr/XXX/}' pipeline/run.py

mkdir dgr_val
mkdir dgr_val/5way5shot/
mkdir dgr_val/5way5shot/CUB
mkdir dgr_val/5way1shot/
mkdir dgr_val/5way1shot/CUB

sed -i '0,/XXX/{s/XXX/val/}' run_ens_checkI.sh
sed -i '0,/XXX/{s/XXX/dgr_val/}' pipeline/run.py
bash run_ens_checkII.sh CUB 5 # DeepVoro 5-way 5-shot val set
bash run_ens_checkII.sh CUB 1 # DeepVoro 5-way 1-shot val set
sed -i '0,/val/{s/val/XXX/}' run_ens_checkI.sh
sed -i '0,/dgr_val/{s/dgr_val/XXX/}' pipeline/run.py
