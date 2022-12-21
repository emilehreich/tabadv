UB_GRID="10000.0"
mkdir ../models/home_credit_log_l1/
for EPS in $UB_GRID
do
	echo "Training"
	echo $EPS
	python ./train.py --dataset home_credit --eps $EPS --model_path ../models/home_credit_log_l1/l1_$EPS.pt --mixed-loss --attack_iters 100 --batch_size 2048 --epochs 50
done
