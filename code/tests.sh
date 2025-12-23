#!/bin/bash

SUBJECT=1
MODES="brain visual"
DATA_DIR="~/Desktop/uni/ml_coursework/data"
K_TEST=10
NUM_SAMPLES=10000

K_NEIGHBORS=15
OUT_DIM=64
MIN_DIST=0.1
TRAIN_EPOCHS=200
NUM_REP=8
LR=0.2
ALPHA=1.0
BATCH_SIZE=32
TEST_EPOCHS=50

mkdir -p losses

echo -e "\nTESTING LR\n"
best_lr=""
best_lr_score=999999

for lr in 0.05 0.1 0.2 0.5; do
    save_dir="losses/lr_${lr}_lr${lr}_alpha${ALPHA}_numrep${NUM_REP}_outdim${OUT_DIM}.npz"
    
    output=$(python3 code/main.py \
        --k_neighbors $K_NEIGHBORS \
        --out_dim $OUT_DIM \
        --min_dist $MIN_DIST \
        --train_epochs $TRAIN_EPOCHS \
        --num_rep $NUM_REP \
        --lr $lr \
        --alpha $ALPHA \
        --batch_size $BATCH_SIZE \
        --num_samples $NUM_SAMPLES \
        --save_dir "$save_dir" \
        --data_dir "$DATA_DIR" \
        --subject $SUBJECT \
        --modes $MODES \
        --test_epochs $TEST_EPOCHS \
        --k_test $K_TEST)
    
    sim_score=$(echo "$output" | grep "Average cross-modal distance:" | awk '{print $4}')
    knn_score=$(echo "$output" | grep "Average KNN accuracy:" | awk '{print $4}')
    
    echo "lr=${lr}: sim=${sim_score}, knn=${knn_score}"
    
    if (( $(echo "$sim_score < $best_lr_score" | bc -l) )); then
        best_lr_score=$sim_score
        best_lr=$lr
    fi
done

echo -e "\nBest lr: ${best_lr} (sim score: ${best_lr_score})\n"

echo -e "\nTESTING ALPHA\n"
best_alpha=""
best_alpha_score=999999

for alpha in 0.5 1.0 2.0 4.0; do
    save_dir="losses/alpha_${alpha}_lr${best_lr}_alpha${alpha}_numrep${NUM_REP}_outdim${OUT_DIM}.npz"
    
    output=$(python3 code/main.py \
        --k_neighbors $K_NEIGHBORS \
        --out_dim $OUT_DIM \
        --min_dist $MIN_DIST \
        --train_epochs $TRAIN_EPOCHS \
        --num_rep $NUM_REP \
        --lr $best_lr \
        --alpha $alpha \
        --batch_size $BATCH_SIZE \
        --num_samples $NUM_SAMPLES \
        --save_dir "$save_dir" \
        --data_dir "$DATA_DIR" \
        --subject $SUBJECT \
        --modes $MODES \
        --test_epochs $TEST_EPOCHS \
        --k_test $K_TEST)
    
    sim_score=$(echo "$output" | grep "Average cross-modal distance:" | awk '{print $4}')
    knn_score=$(echo "$output" | grep "Average KNN accuracy:" | awk '{print $4}')
    
    echo "alpha=${alpha}: sim=${sim_score}, knn=${knn_score}"
    
    if (( $(echo "$sim_score < $best_alpha_score" | bc -l) )); then
        best_alpha_score=$sim_score
        best_alpha=$alpha
    fi
done

echo -e "\nBest alpha: ${best_alpha} (sim score: ${best_alpha_score})\n"

echo -e "\nTESTING NUM_REP\n"
best_num_rep=""
best_num_rep_score=999999

for num_rep in 4 8 16; do
    save_dir="losses/num_rep_${num_rep}_lr${best_lr}_alpha${best_alpha}_numrep${num_rep}_outdim${OUT_DIM}.npz"
    
    output=$(python3 code/main.py \
        --k_neighbors $K_NEIGHBORS \
        --out_dim $OUT_DIM \
        --min_dist $MIN_DIST \
        --train_epochs $TRAIN_EPOCHS \
        --num_rep $num_rep \
        --lr $best_lr \
        --alpha $best_alpha \
        --batch_size $BATCH_SIZE \
        --num_samples $NUM_SAMPLES \
        --save_dir "$save_dir" \
        --data_dir "$DATA_DIR" \
        --subject $SUBJECT \
        --modes $MODES \
        --test_epochs $TEST_EPOCHS \
        --k_test $K_TEST)
    
    sim_score=$(echo "$output" | grep "Average cross-modal distance:" | awk '{print $4}')
    knn_score=$(echo "$output" | grep "Average KNN accuracy:" | awk '{print $4}')
    
    echo "num_rep=${num_rep}: sim=${sim_score}, knn=${knn_score}"
    
    if (( $(echo "$sim_score < $best_num_rep_score" | bc -l) )); then
        best_num_rep_score=$sim_score
        best_num_rep=$num_rep
    fi
done

echo -e "\nBest num_rep: ${best_num_rep} (sim score: ${best_num_rep_score})\n"

echo -e "\nTESTING OUT_DIM\n"

for out_dim in 64 128 192; do
    save_dir="losses/out_dim_${out_dim}_lr${best_lr}_alpha${best_alpha}_numrep${best_num_rep}_outdim${out_dim}.npz"
    
    output=$(python3 code/main.py \
        --k_neighbors $K_NEIGHBORS \
        --out_dim $out_dim \
        --min_dist $MIN_DIST \
        --train_epochs $TRAIN_EPOCHS \
        --num_rep $best_num_rep \
        --lr $best_lr \
        --alpha $best_alpha \
        --batch_size $BATCH_SIZE \
        --num_samples $NUM_SAMPLES \
        --save_dir "$save_dir" \
        --data_dir "$DATA_DIR" \
        --subject $SUBJECT \
        --modes $MODES \
        --test_epochs $TEST_EPOCHS \
        --k_test $K_TEST)
    
    sim_score=$(echo "$output" | grep "Average cross-modal distance:" | awk '{print $4}')
    knn_score=$(echo "$output" | grep "Average KNN accuracy:" | awk '{print $4}')
    
    echo "out_dim=${out_dim}: sim=${sim_score}, knn=${knn_score}"
done

echo -e "\nFINAL SUMMARY\n"
echo "Best LR: ${best_lr}"
echo "Best alpha: ${best_alpha}"
echo "Best num_rep: ${best_num_rep}"
