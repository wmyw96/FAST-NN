for p in 4000 5000
do
		for s in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
		do
			echo $p $s
			python far_exp.py --p $p --seed $s --exp_id 1 --lr 1e-3 --record_dir logs/exp1-lre-3
		done
done