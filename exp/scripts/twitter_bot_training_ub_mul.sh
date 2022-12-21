UB_MUL_GRID="0.001 0.003 0.01 0.03 0.1 0.3 1.0"
mkdir ../models/twitter_bot/
for EPS in $UB_MUL_GRID
do
	echo "Training"
	echo $EPS
	python train.py --dataset twitter_bot --eps $EPS --model_path ../models/twitter_bot/ub_mul_$EPS.pt --eps-sched --utility-type multiplicative --attack_iters 10 --batch_size 2048 --epochs 20
done
