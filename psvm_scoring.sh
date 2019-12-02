#!/bin/bash
# Copyright 2019   Jerry Peng
# Apache 2.0.
#
# This script does PSVM pairwise scoring.
# In the future, standard scoring will be added.

# TODO: Add comments

# Begin configuartion section.
num_threads=10
cmd="run.pl"
stage=
ivec_num=false
ivec_center=false

export OPENBLAS_NUM_THREADS=${num_threads} # this is the number of threads to do Matrix multiplication

# End configuartion section.

echo "$0 $@" # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
    echo "Usage: $0 <psvm-dir> <test-ivector-dir> <trials> <scores>"
    echo " e.g.: $0 exp/psvm exp/ivectors_test data/voxceleb1_trials_sv exp/psvm/scores/psvm.score"
    echo "main options (for other, see top of script file)"
    echo "  --cmd (utils/run.pl/utils/queue.pl <queue opts> # how to run jobs."
    echo "  --num-threads <n|10>                            # Number of threads for matrix multiplication"
    echo "                                                  # OpenBLAS should be used in this case"
    echo "  --stage <stage|>                                # To control partial reruns"
    echo "  --ivec-norm <true,false|false>                  # To normalize ivectors before training"
    echo "  --ivec-center <true,false|false>                # To center ivectors before training"
    exit 1;
fi

psvm_dir=$1
ivec_dir=$2
trials=$3
scores=$4

mkdir -p $psvm_dir/log

for f in $psvm_dir/final.psvm $ivec_dir/ivector.scp $trials; do
    [ ! -f $f ] && echo "No such file $f" && exit 1;
done

ivecs="scp:$ivec_dir/ivector.scp"
if $ivec_norm && ! $ivec_center; then
    ivecs="ark:ivector-normalize-length $ivecs ark:- |"
fi
if ! $ivec_norm && $ivec_center; then
    [ ! -f $psvm_dir/mean.vec ] && echo "No such file $f" && exit 1;
    ivecs="ark:ivector-subtract-global-mean $psvm_dir/mean.vec $ivecs ark:- |"
fi
if $ivec_norm && $ivec_center; then
    [ ! -f $psvm_dir/mean.vec ] && echo "No such file $f" && exit 1;
    ivecs="ark:ivector-normalize-length $ivecs ark:- | ivector-subtract-global-mean $psvm_dir/mean.vec ark:- ark:- | ivector-normalize-length ark:- ark:- |"
fi

$cmd $psvm_dir/log/psvm_pairwise_scoring.log \
    psvm-pairwise-scoring \
        $psvm_dir/final.psvm \
        "$ivecs" \
        "cat '$trials' | cut -d\  --fields=1,2 |" \
        $scores || exit 1;

