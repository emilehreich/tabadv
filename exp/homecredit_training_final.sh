GRID="10.0 20.0 200.0 400.0 800.0 1000.0 3000.0 10000.0"
#GRID="200.0 400.0 1000.0 2000.0"
#GRID=""
for EPS in $GRID
do
	echo "Training"
	echo $EPS
	python train.py --dataset home_credit --eps $EPS --model_path ../models/home_credit/l1_40_$EPS.pt --eps-sched --same-cost --mixed-loss --attack_iters 40 --batch_size 512 --epochs 40
done
UB_GRID=""
for EPS in $UB_GRID
do
	echo "Training"
	echo $EPS
	python train.py --dataset home_credit --eps $EPS --model_path ../models/homecredit_final/ub_$EPS.pt --eps-sched  --mixed-loss --utility-type additive --attack_iters 20 --batch_size 2048 --epochs 200
done
