// psvm/psvm.h
// Copyright 2019 Jerry Peng
//
//
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


#ifndef KALDI_PSVM_H_
#define KALDI_PSVM_H_

#include <vector>
#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "itf/options-itf.h"
#include "util/common-utils.h"

namespace kaldi {
/* This code implements Pariwise Support Vector Machine: see
 * "Large-Scale Training of Pairwise Support Vector Machines 
 * for Speaker Recognition" by Sandro Cumani, IEEE TASLP 2014.
 * At least, that was the insipration. The solver is based on
 * Stochastic Gradient Descent. For the details, search Pegasos.
 */

struct PsvmPair{
    PsvmPair(int32 i, int32 j, float label): i(i), j(j), label(label) { }
    PsvmPair() {}
    int32 i;
    int32 j;
    float label;
};


    // std::vector<PsvmPair> psvmPairVec;

void psvmPairVecWrite(std::ostream &os, bool binary, const std::vector<PsvmPair> &psvmPairVec) {
    size_t num = psvmPairVec.size();
    KALDI_ASSERT(num > 0);
    WriteToken(os, binary, "<Pairs>");
    WriteBasicType(os, binary, num);
    for (size_t i=0; i < num; ++i) {
        const PsvmPair &pair = psvmPairVec[i];
        if (false == binary)
            os.put('\n');
        WriteBasicType(os, binary, pair.i);
        WriteBasicType(os, binary, pair.j);
        WriteBasicType(os, binary, pair.label);
    }
    WriteToken(os, binary, "</Pairs>");
}

void psvmPairVecRead(std::istream &is, bool binary, std::vector<PsvmPair> &psvmPairVec) {
    ExpectToken(is, binary, "<Pairs>");
    size_t num;
    ReadBasicType(is, binary, &num);
    if (num < 1) {
        KALDI_ERR << "Error reading Pair vector: size = "
                  << num;
    }
    psvmPairVec.resize(num);
    for (size_t i=0; i < num; ++i) {
        PsvmPair &pair = psvmPairVec[i];
        if (false == binary)
            is.get(); // eat up '/n'
        ReadBasicType(is, binary, &(pair.i));
        ReadBasicType(is, binary, &(pair.j));
        ReadBasicType(is, binary, &(pair.label));
    }
    ExpectToken(is, binary, "</Pairs>");
}

class Psvm {
    public:
        Psvm() { }

        explicit Psvm(const Psvm &other):
            Lambda_(other.Lambda_),
            Gamma_(other.Gamma_),
            c_(other.c_),
            k_(other.k_) {
        };

        void init(bool isRandom, int32 dim);

        void precompute(const Matrix<double> *pivecs);

        float scoring(const int32 idx1, const int32 idx2) const;

        int32 Dim() const { return Lambda_.NumRows(); }

        int32 NumIvec() const { return Lambda_x_.NumRows(); }

        void Write(std::ostream &os, bool binary) const;

        void Read(std::istream &is, bool binary);
//    protected: //for the easy use of python
        friend class PsvmEstimator;
        
        SpMatrix<double> Lambda_;
        SpMatrix<double> Gamma_;
        Vector<double> c_;
        double k_;
        
        // intermediate results produced by precompute
        Matrix<double> Lambda_x_;
        Vector<double> xt_Gamma_x_;
        Vector<double> xt_c_;
        const Matrix<double> *px_;

    private:
        Psvm &operator = (const Psvm &other); // disallow assignment
};

// struct PsvmEstimatorSequentialConfig{
//     // This config is for the sequential training of Pairwsie-SVM
//     bool random_init;
//     // size_t batch_size;
//     // float penalty;
//     // float neg_pos_ratio;
//     // float lr; // it will be an input arg in func: UpdateParas()
//     // PsvmEstimatorSequentialConfig(): random_init(true), batch_size(1000000),
//     //    penalty(10.0), neg_pos_ratio(10.0) { }
//     PsvmEstimatorSequentialConfig(): random_init(true) { }
//     void Register(OptionsItf *opts) {
//         opts->Register("random-init", &random_init,
//                 "If true, randomly initialize Psvm parameters; "
//                 "Otherwise, set all parameters to zero.");
//         // opts->Register("batch-size", &batch_size,
//         //        "The number of pairs for training one iteration "
//         //        "1000000 by default.");
//         // opts->Register("penalty", &penalty,
//         //         "The penalty for clf-loss, this is to adjust the"
//         //         " weight between parameter loss and classification loss.");
//         // opts->Register("neg-pos-ratio", &neg_pos_ratio,
//         //         "The ratio of #neg-pairs over #pos-pairs. "
//         //         "It is used to handle the unbalanced data pairs.");
//     } // for the flexibilty of adjusting learning rate, I provide an input arg of lr
//     // in UpdateParas() rather than register it here
// 
// };

class PsvmEstimator{
    public:
        // InitPsvmParas()
        // ResetStats()
        PsvmEstimator(bool random_init,
                      const Matrix<double> *pivecs);
        PsvmEstimator(Psvm *ppsvm, const Matrix<double> *pivecs);
        // void InitPsvmParas(bool isRandom) const;
        void ResetStats();
        void Forward(const std::vector<PsvmPair> &psvmPairVec,
                const size_t start_idx, const size_t end_idx, float pos_weight);
        void Backward(double lr, double update_weight) const;
        double GetClfLoss() const;
        double GetParaLoss() const;
        void OneIter(const std::vector<PsvmPair> &psvmPairVec,
                     const size_t start_idx,
                     const size_t end_idx,
                     double neg_pos_ratio,
                     double lr,
                     double penalty);

        int32 DimIvec() const {return pivecs_->NumCols();}
        int32 NumIvec() const {return pivecs_->NumRows();}

//    private:
        const Matrix<double> *pivecs_;
        Psvm *ppsvm_;

        Matrix<double> stats_1th_;
        Vector<double> stats_0th_row_;
        Vector<double> stats_0th_col_;
        double clf_loss_; // classifiation loss
        size_t sv_num_; // number of support vectors

        KALDI_DISALLOW_COPY_AND_ASSIGN(PsvmEstimator);
};

} // namespace kaldi

#endif

