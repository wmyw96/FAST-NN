for p in 300 400 600 700 900
do
		for s in 8
		do
			echo $p $s
			python far_exp.py --p $p --seed $s
		done
done