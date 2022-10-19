for p in 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000
do
		for s in {0..199}
		do
			echo $p $s
			python far_exp.py --p $p --seed $s --exp_id 2 --dropout_rate 0.3 --record_dir logs/exp2-0.3
		done
done