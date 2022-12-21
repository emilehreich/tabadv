UB_GRID="200000.0 300000.0"
mkdir ../models/home_credit_log_20/
for EPS in $UB_GRID
do
	echo "Training"
	echo $EPS
	python ./train.py --dataset home_credit --keep-one-hot --eps $EPS --model_path ../models/home_credit_log_20/ub_$EPS.pt --attack_iters 20 --batch_size 2048 --epochs 100 --utility-type additive
done
