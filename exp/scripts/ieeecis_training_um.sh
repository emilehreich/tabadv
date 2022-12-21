#UB_GRID="0.0 10.0 20.0 50.0 100.0 200.0 500.0"
UB_GRID="0.0003 0.001 0.003"
mkdir ../models/ieeecis/
for EPS in $UB_GRID
do
	echo "Training"
	echo $EPS
	python train.py --dataset ieeecis --lamb $EPS --utility-max --model_path ../models/ieeecis/um_$EPS.pt --mixed-loss --utility-type additive --keep-one-hot --attack_iters 10 --batch_size 2048 --epochs 400
done
