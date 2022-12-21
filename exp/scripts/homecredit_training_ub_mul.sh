UB_GRID="0.00001 0.00003 0.0001 0.0003 0.001 0.003 0.01 0.1 1.0"
mkdir ../models/home_credit/
for EPS in $UB_GRID
do
	echo "Training"
	echo $EPS
	python train.py --dataset home_credit --eps $EPS --model_path ../models/home_credit/ub_mul_$EPS.pt --eps-sched --utility-type multiplicative --attack_iters 10 --batch_size 2048 --epochs 200
done
