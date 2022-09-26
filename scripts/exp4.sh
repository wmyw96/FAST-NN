for p in 5000
do
    for s in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
    do
        echo $p $s
        python fast_exp.py --p $p --seed $s --hcm_id 3 --record_dir logs/exp4-hcm3-m200 --m 200 --hp_tau 0.005
    done
done
