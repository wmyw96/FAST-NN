for p in 100 200 500 800 1000 2000 3000 4000 5000 6000
do
		for s in 0 1 2 3 4 5 6 7 8 9
		do
			echo $p $s
			python far_exp.py --p $p --seed $s
		done
done