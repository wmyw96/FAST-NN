for p in 100 500 1000 5000
do
    for k in 3 5 8 50
    do
        for s in {0..200}
        do
          echo $p $s
          python far_exp.py --p $p --seed $s --exp_id 3 --m $k --record_dir logs/exp3
        done
    done
done
