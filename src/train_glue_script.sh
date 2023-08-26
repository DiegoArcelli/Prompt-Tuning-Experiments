if [ $# -eq 0 ]
    then
        mode="normal"
        lr=2e-5
        bsize=2
        epochs=10
    else
        mode=$1
        lr=$2
        bsize=$3
        epochs=$4
fi

python3 train_roberta_glue_task.py -t cola -b $bsize -lr $lr -m $mode -e $epochs > terminal_output/output_glue_cola_$mode
python3 train_roberta_glue_task.py -t mnli -b $bsize -lr $lr -m $mode -e $epochs > terminal_output/output_glue_cola_$mode
python3 train_roberta_glue_task.py -t mrpc -b $bsize -lr $lr -m $mode -e $epochs > terminal_output/output_glue_cola_$mode
python3 train_roberta_glue_task.py -t qnli -b $bsize -lr $lr -m $mode -e $epochs > terminal_output/output_glue_cola_$mode
python3 train_roberta_glue_task.py -t qqp -b $bsize -lr $lr -m $mode -e $epochs > terminal_output/output_glue_cola_$mode
python3 train_roberta_glue_task.py -t rte -b $bsize -lr $lr -m $mode -e $epochs > terminal_output/output_glue_cola_$mode
python3 train_roberta_glue_task.py -t sst2 -b $bsize -lr $lr -m $mode -e $epochs > terminal_output/output_glue_cola_$mode
python3 train_roberta_glue_task.py -t wnli -b $bsize -lr $lr -m $mode -e $epochs > terminal_output/output_glue_cola_$mode

