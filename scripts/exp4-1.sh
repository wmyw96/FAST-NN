for s in {0..199}
do
    for p in 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000
    do
        echo $s $p
        python fast_exp.py --p $p --seed $s --hcm_id 0 --record_dir logs/exp4-hcm0-m200 --m 200
    done
done
