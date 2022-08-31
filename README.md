# Simple, Scalable Adaptation for Neural Machine Translation

Implementation of paper ["Simple, Scalable Adaptation for Neural Machine Translation"](https://arxiv.org/abs/1909.08478). 

## Train a baseline transformer model
Environment variable. Create a file `.env` and `source .env`.
```
FAIRSEQ_SRC=/path/to/fairseq
RAW_DATA_DIR=raw
DATA_DIR=data
DATA_BIN=databin
CHECKPOINTS=checkpoints
CUDA_VISIBLE_DEVICES=0
MAX_TOKENS=4096  # decide by gpu memory
```

Download wmt18 en-zh processed training data
```
curl -o $RAW_DATA_DIR/corpus.gz  https://data.statmt.org/wmt18/translation-task/preprocessed/zh-en/corpus.gz
gunzip $RAW_DATA_DIR/corpus.gz
cut -f 1 $DATA_DIR/corpus > en.train
cut -f 2 $DATA_DIR/corpus > zh.train

sacrebleu -t wmt17 -l en-zh --echo src > $RAW_DATA_DIR/en.valid
sacrebleu -t wmt17 -l en-zh --echo ref > $RAW_DATA_DIR/zh.valid
sacrebleu -t wmt18 -l en-zh --echo src > $RAW_DATA_DIR/en.test
sacrebleu -t wmt18 -l en-zh --echo ref > $RAW_DATA_DIR/zh.test
```
Train sentencepiece model and tokenize data
```
python $FAIRSEQ_SRC/scripts/spm_train.py \
    --input=$RAW_DATA_DIR/en.train,$RAW_DATA_DIR/zh.train \
    --model_prefix=$DATA_DIR/sentencepiece.bpe \
    --vocab_size=32000 \
    --character_coverage=1.0 \
    --model_type=bpe \
    --input_sentence_size=2000000 \
    --shuffle_input_sentence=true

python $FAIRSEQ_SRC/scripts/spm_encode.py \
    --model $DATA_DIR/sentencepiece.bpe.model \
    --inputs $RAW_DATA_DIR/en.train $RAW_DATA_DIR/zh.train \
    --outputs $DATA_DIR/train.en $DATA_DIR/train.zh

python $FAIRSEQ_SRC/scripts/spm_encode.py \
    --model $DATA_DIR/sentencepiece.bpe.model \
    --inputs $RAW_DATA_DIR/en.valid $RAW_DATA_DIR/zh.valid \
    --outputs $DATA_DIR/valid.en $DATA_DIR/valid.zh

python $FAIRSEQ_SRC/scripts/spm_encode.py \
    --model $DATA_DIR/sentencepiece.bpe.model \
    --inputs $RAW_DATA_DIR/en.test $RAW_DATA_DIR/zh.test \
    --outputs $DATA_DIR/test.en $DATA_DIR/test.zh
```
Binarize data
```
fairseq-preprocess --source-lang en --target-lang zh \
    --trainpref $DATA_DIR/train \
    --validpref $DATA_DIR/valid \
    --testpref $DATA_DIR/test \
    --joined-dictionary \
    --thresholdtgt 10 --thresholdsrc 10 \
    --destdir $DATA_BIN \
    --workers 40 
```
Train normal transformer model without adapters.
```
fairseq-train \
    $RAW_DATA_DIR \
    --arch transformer --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.1 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens $MAX_TOKENS \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
```
Evalueate bleu score on valid and test data.
```
fairseq-generate $DATA_BIN \
    --path checkpoints/checkpoint_best.pt \
    --beam 5 --remove-bpe | \
    --gen-subset valid \
    | grep ^H | LC_ALL=C sort -V | cut -f3- | 
    sacrebleu -t wmt18 -l en-zh

fairseq-generate $DATA_BIN \
    --path checkpoints/checkpoint_best.pt \
    --beam 5 --remove-bpe | \
    --gen-subset test \
    | grep ^H | LC_ALL=C sort -V | cut -f3- | 
    sacrebleu -t wmt18 -l en-zh
```
## Fine-tune adapter layers



## Run as an server
