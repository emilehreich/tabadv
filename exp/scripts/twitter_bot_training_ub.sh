UB_GRID="-1000 -100.0 -10.0 0.0 10.0 20.0"
mkdir ../models/twitter_bot/
for EPS in $UB_GRID
do
	echo "Training"
	echo $EPS
	python train.py --dataset twitter_bot --eps $EPS --model_path ../models/twitter_bot_st/ub_$EPS.pt --eps-sched --utility-type additive --attack_iters 10 --batch_size 128 --epochs 20
done
