set -euo pipefail

# PYTHON="conda run --no-capture-output -n linux_env python"
PYTHON="python"
OUT="results"
DATASET="${DATASET:-cifar10}"   # override with: DATASET=synthetic bash run_all.sh


echo "=== Pretraining teachers ==="

for SEED in 0 42; do
  $PYTHON train.py --baseline b0 --teacher rnn --student rnn \
    --dataset $DATASET --seed $SEED --out_dir $OUT
  $PYTHON train.py --baseline b0 --teacher cnn --student cnn \
    --dataset $DATASET --seed $SEED --out_dir $OUT
done

RNN_CKPT_S0="$OUT/b0_rnn_rnn_seed0/best.pt"
RNN_CKPT_S42="$OUT/b0_rnn_rnn_seed42/best.pt"
CNN_CKPT_S0="$OUT/b0_cnn_cnn_seed0/best.pt"
CNN_CKPT_S42="$OUT/b0_cnn_cnn_seed42/best.pt"


echo "=== B0: No distillation ==="

for SEED in 0 42; do
  $PYTHON train.py --baseline b0 --teacher rnn --student cnn \
    --dataset $DATASET --seed $SEED --out_dir $OUT
  $PYTHON train.py --baseline b0 --teacher cnn --student rnn \
    --dataset $DATASET --seed $SEED --out_dir $OUT
done


echo "=== B1: Response-based KD ==="

for SEED in 0 42; do
  TCKPT=$([ $SEED -eq 0 ] && echo $RNN_CKPT_S0 || echo $RNN_CKPT_S42)
  $PYTHON train.py --baseline b1 --teacher rnn --student cnn \
    --teacher_ckpt $TCKPT --dataset $DATASET --seed $SEED --out_dir $OUT

  TCKPT=$([ $SEED -eq 0 ] && echo $CNN_CKPT_S0 || echo $CNN_CKPT_S42)
  $PYTHON train.py --baseline b1 --teacher cnn --student rnn \
    --teacher_ckpt $TCKPT --dataset $DATASET --seed $SEED --out_dir $OUT
done


echo "=== B2: Hard fixed alignment ==="

for SEED in 0 42; do
  TCKPT=$([ $SEED -eq 0 ] && echo $RNN_CKPT_S0 || echo $RNN_CKPT_S42)
  $PYTHON train.py --baseline b2 --teacher rnn --student cnn \
    --teacher_ckpt $TCKPT --dataset $DATASET --seed $SEED --out_dir $OUT

  TCKPT=$([ $SEED -eq 0 ] && echo $CNN_CKPT_S0 || echo $CNN_CKPT_S42)
  $PYTHON train.py --baseline b2 --teacher cnn --student rnn \
    --teacher_ckpt $TCKPT --dataset $DATASET --seed $SEED --out_dir $OUT
done


echo "=== B3: Uniform alignment ==="

for SEED in 0 42; do
  TCKPT=$([ $SEED -eq 0 ] && echo $RNN_CKPT_S0 || echo $RNN_CKPT_S42)
  $PYTHON train.py --baseline b3 --teacher rnn --student cnn \
    --teacher_ckpt $TCKPT --dataset $DATASET --seed $SEED --out_dir $OUT

  TCKPT=$([ $SEED -eq 0 ] && echo $CNN_CKPT_S0 || echo $CNN_CKPT_S42)
  $PYTHON train.py --baseline b3 --teacher cnn --student rnn \
    --teacher_ckpt $TCKPT --dataset $DATASET --seed $SEED --out_dir $OUT
done


echo "=== Proposed method ==="

for SEED in 0 42; do
  TCKPT=$([ $SEED -eq 0 ] && echo $RNN_CKPT_S0 || echo $RNN_CKPT_S42)
  $PYTHON train.py --baseline proposed --teacher rnn --student cnn \
    --teacher_ckpt $TCKPT --tau 1.0 --direction row \
    --dataset $DATASET --seed $SEED --out_dir $OUT

  TCKPT=$([ $SEED -eq 0 ] && echo $CNN_CKPT_S0 || echo $CNN_CKPT_S42)
  $PYTHON train.py --baseline proposed --teacher cnn --student rnn \
    --teacher_ckpt $TCKPT --tau 1.0 --direction row \
    --dataset $DATASET --seed $SEED --out_dir $OUT
done


echo "=== B5: Direction swap ==="

for SEED in 0 42; do
  for DIRECTION in row col; do
    TCKPT=$([ $SEED -eq 0 ] && echo $RNN_CKPT_S0 || echo $RNN_CKPT_S42)
    $PYTHON train.py --baseline proposed --teacher rnn --student cnn \
      --teacher_ckpt $TCKPT --tau 1.0 --direction $DIRECTION \
      --dataset $DATASET --seed $SEED \
      --out_dir "${OUT}/b5_rnn_cnn_tau1.0_dir${DIRECTION}_seed${SEED}"

    TCKPT=$([ $SEED -eq 0 ] && echo $CNN_CKPT_S0 || echo $CNN_CKPT_S42)
    $PYTHON train.py --baseline proposed --teacher cnn --student rnn \
      --teacher_ckpt $TCKPT --tau 1.0 --direction $DIRECTION \
      --dataset $DATASET --seed $SEED \
      --out_dir "${OUT}/b5_cnn_rnn_tau1.0_dir${DIRECTION}_seed${SEED}"
  done
done


echo "=== Generating figures ==="
$PYTHON visualize.py --results_dir $OUT --out_dir figures/

echo "=== All done. Results in $OUT/, figures in figures/ ==="
