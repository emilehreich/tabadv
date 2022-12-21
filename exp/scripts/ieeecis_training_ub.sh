UB_GRID="0.0 10.0 20.0 50.0 100.0 200.0 500.0"
#UB_GRID="300.0 400.0"
#UB_GRID="1000.0 2000.0 5000.0"
mkdir ../models/ieeecis/
for EPS in $UB_GRID
do
	echo "Training"
	echo $EPS
	python train.py --dataset ieeecis --eps $EPS --model_path ../models/ieeecis/ub_$EPS.pt --eps-sched --utility-type additive --attack_iters 20 --mixed-loss --batch_size 2048 --epochs 400
done
