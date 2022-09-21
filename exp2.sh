for p in 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000
do
		for s in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
		do
			echo $p $s
			python far_exp.py --p $p --seed $s --exp_id 2 --dropout_rate 0.9 --record_dir logs/exp2-0.9
		done
done