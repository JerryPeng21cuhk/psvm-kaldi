#!/bin/bash

# Copyright 2019 Jerry Peng
# Apache 2.0.

# TODO: Add comments

# Begin configuration section.
use_existing_models=false
num_threads=10
cmd="run.pl"
stage=
num_epochs=2 #to be determined
cleanup=false #will be true after debug
ivec_norm=false
ivec_center=false
batch_size=1000000
neg_pos_ratio=10.0
init_lr=0.3
penalty=3200.0

export OPENBLAS_NUM_THREADS=${num_threads} # this is the number of threads to do Matrix multiplication
# End configuration section.

echo "$0 $@" # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
    echo "Usage: $0 <train-data-dir> <train-ivector-dir> <psvm-dir>"
    echo " e.g.: $0 data/train exp/ivectors_train exp/psvm"
    echo "main options (for other, see top of script file)"
    echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
    echo "  --num-epochs <#epochs|2>                         # Number of epochs of SGD"
    echo "  --num-threads <n|10>                              # Number of threads for matrix multiplication"
    echo "                                                   #  OpenBLAS should be used in this case"
    echo "  --stage <stage|>                                 # To control partial reruns"
    echo "  --ivec-norm <true,false|false>                   # To normalize ivectors before training"
    echo "  --ivec-center <true,false|false>                 # To center ivectors before training"
    echo "  --batch-size <n|1000000>                         # Batch size for SGD training"
    echo "  --neg-pos-ratio <float|10.0>                     # The ratio of #pairs belong to different speakers"
    echo "                                                   #  over #pairs belong to the same speaker"
    echo "  --init-lr <float|0.3>                            # Initial learning rate for SGD"
    echo "  --penalty <float|3200.0>                         # The penalty for classification, this is to adjust the parameter updation step";
    exit 1;
fi

data=$1
ivec_dir=$2
psvm_dir=$3

mkdir -p $psvm_dir/log

if [ "$use_existing_models" == "true" ]; then
    [ ! -f $psvm_dir/final.psvm ] && echo "No such file $f" && exit 1;
    cp $psvm_dir/final.psvm $psvm_dir/0.psvm && mv $psvm_dir/final.psvm $psvm_dir/.final.psvm;
else
    $cmd $psvm_dir/log/psvm_init.log \
        psvm-init --random-init=false $psvm_dir/0.psvm || exit 1;
fi

for f in $data/spk2utt $ivec_dir/ivector.scp ; do
    [ ! -f $f ] && echo "No such file $f" && exit 1;
done

# ivecs="scp:$ivec_dir/ivector.scp"
# if $ivec_norm; then
#     ivecs="ark:ivector-normalize-length $ivecs ark:- |"
#     if $ivec_center; then
#         $cmd $psvm_dir/log/mean.log \
#             ivector-mean $ivecs $psvm_dir/mean.vec || exit 1;
# 
#         ivecs="$ivecs ivector-subtract-global-mean $psvm_dir/mean.vec ark:- ark:- | ivector-normalize-length ark:- ark:- |"
#     fi
# fi

ivecs="scp:$ivec_dir/ivector.scp"
if $ivec_norm && ! $ivec_center; then
    ivecs="ark:ivector-normalize-length $ivecs ark:- |"
fi
if ! $ivec_norm && $ivec_center; then
    $cmd $psvm_dir/log/mean.log \
        ivector-mean $ivecs $psvm_dir/mean.vec || exit 1;
    ivecs="ark:ivector-subtract-global-mean $psvm_dir/mean.vec $ivecs ark:- |"
fi
if $ivec_norm && $ivec_center; then
    $cmd $psvm_dir/log/mean.log \
        ivector-mean "ark:ivector-normalize-length $ivec ark:- |" $psvm_dir/mean.vec || exit 1;
    ivecs="ark:ivector-normalize-length $ivecs ark:- | ivector-subtract-global-mean $psvm_dir/mean.vec ark:- ark:- | ivector-normalize-length ark:- ark:- |"
fi


$cmd $psvm_dir/log/psvm_generate_pair.log \
    psvm-generate-pairs --neg-pos-ratio=$neg_pos_ratio \
        ark:$data/spk2utt \
        scp:$ivec_dir/ivector.scp \
        $psvm_dir/train_pairs || exit 1;

for epoch in $(seq 1 $num_epochs); do
    $cmd $psvm_dir/log/psvm_est.log \
        psvm-est --neg-pos-ratio=$neg_pos_ratio \
            --init-lr=$init_lr \
            --batch-size=$batch_size \
            --penalty=$penalty \
            $psvm_dir/$[$epoch-1].psvm \
            $psvm_dir/train_pairs \
            "$ivecs" \
            $psvm_dir/${epoch}.psvm || exit 1;
    $cleanup && rm $psvm_dir/$[$epoch-1].psvm;
done

$cleanup && rm $psvm_dir/final.psvm
ln -s ${num_epochs}.psvm $psvm_dir/final.psvm

