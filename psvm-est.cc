// psvmbin/psvm-est.cc

// Copyright 2019 Jerry Peng

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "psvm/psvm.h"

int main(int argc, char *argv[]) {
    using namespace kaldi;
    try {
        const char *usage =
            "Do model estimation of psvm extractor\n"
            "If psvm-in is given, do model re-estimation\n"
            "Usage: psvm-est [options] <psvm-in> <train-pairs-rxfilename> <ivector-rspecifier> <psvm-out>\n"
            "e.g.:\n"
            " psvm-est psvm-0.mdl exp/psvm/train_pairs scp:exp/ivectors_train/ivectors.scp psvm.mdl\n";

        bool binary = true;
        double init_lr = 0.03;
        unsigned int batch_size = 1000000;
        double neg_pos_ratio = 10.0;
        double penalty = 10.0;
        
        ParseOptions po(usage);
        po.Register("init-lr", &init_lr, "Inital learning rate for the SGD training of PSVM. (0.03 by default)");
        po.Register("batch-size", &batch_size, "The number of training pairs for one iteration. (1000000 by default)");
        po.Register("neg-pos-ratio", &neg_pos_ratio, "The ratio of #neg-pairs over #pos-pairs.");
        po.Register("penalty", &penalty, "The penaltyfor classification loss, this is to balance between parameter loss and classification loss.");
        po.Register("binary", &binary, "Write output in binary mode");

        po.Read(argc, argv);

        if (po.NumArgs() != 4) {
            po.PrintUsage();
            exit(1);
        }

        std::string psvm_rxfilename = po.GetArg(1),
                    train_pair_rxfilename = po.GetArg(2),
                    train_ivector_rspecifier = po.GetArg(3),
                    psvm_wxfilename = po.GetArg(4);

        Matrix<double> ivector_mat; //(ivectors.size(), ivectors[0].Dim());
        {
            // load ivector
            KALDI_LOG << "Loading training ivectors from " << train_ivector_rspecifier;
            SequentialBaseFloatVectorReader ivector_reader(train_ivector_rspecifier);
            std::vector<Vector<BaseFloat> > ivectors;
            for (; !ivector_reader.Done(); ivector_reader.Next()) {
                std::string uttid = ivector_reader.Key();
                ivectors.push_back(ivector_reader.Value());
            }
            if (ivectors.size() == 0) {
                KALDI_ERR << "No ivector exists!";
            }
            ivector_mat.Resize(ivectors.size(), ivectors[0].Dim());
            for (size_t i = 0; i < ivectors.size(); i++)
                ivector_mat.Row(i).CopyFromVec(ivectors[i]);
        }

        std::vector<PsvmPair> pairVec;
        {
            KALDI_LOG << "Loading training pairs from " << train_pair_rxfilename;
            // load pairs for training
            bool binary_in;
            Input psvmPairReader(train_pair_rxfilename, &binary_in);
            psvmPairVecRead(psvmPairReader.Stream(), binary_in, pairVec);
        }
        
        Psvm psvm;
        KALDI_LOG << "Loading Pairwise SVM from " << psvm_rxfilename;
        ReadKaldiObject(psvm_rxfilename, &psvm);
        KALDI_LOG << "Start training...";
        PsvmEstimator psvm_est(&psvm, &ivector_mat);

        int num_iter = pairVec.size() / batch_size;
        if (num_iter < 1) {
            KALDI_ERR << "Batch size " << batch_size << " is too large, "
                << "which leads to less than one iteration for SGD.";
        }
        for (size_t i=0; i < num_iter; i++) {
            double lr = init_lr / (i+1);
            size_t start_idx = i * batch_size;
            size_t end_idx = start_idx + batch_size;
            psvm_est.OneIter(pairVec, start_idx, end_idx, neg_pos_ratio, lr, penalty);
        }

        WriteKaldiObject(psvm, psvm_wxfilename, binary);
        KALDI_LOG << "Finish PSVM training and wrote it to " << psvm_wxfilename;
        return 0;
    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}
