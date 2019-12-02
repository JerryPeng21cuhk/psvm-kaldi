// psvm/psvm.c
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


#include "psvm/psvm.h"
#include "base/timer.h"

namespace kaldi{

static void RandFullCova(SpMatrix<double>* matrix) {
    size_t dim = matrix->NumCols();
    KALDI_ASSERT(matrix->NumCols() == matrix->NumRows());

    size_t iter = 0;
    size_t max_iter = 10000;
    // generate random (non-singular) matrix
    // until condition
    Matrix<BaseFloat> tmp(dim, dim);
    SpMatrix<BaseFloat> tmp2(dim);
    while (iter < max_iter) {
      tmp.SetRandn();
      if (tmp.Cond() / dim < 5) break;
      iter++;
    }
    if (iter >= max_iter) {
      KALDI_ERR << "Internal error: found no random covariance matrix.";
    }
    // tmp * tmp^T will give positive definite matrix
    tmp2.AddMat2(1.0, tmp, kNoTrans, 0.0);
    matrix->CopyFromSp(tmp2);
}

// AddMatMat is too slow comparing to np.dot (4x slower)
void EfficientMatMul(Matrix<double> &dest, const Matrix<double> &A, const Matrix<double> &B) {
    KALDI_ASSERT(A.NumCols() == B.NumRows() &&
                 A.NumRows() == dest.NumRows() &&
                 B.NumCols() == dest.NumCols() &&
                 "KALDI_ERR: EfficientMatMul: arguments have mismatched dimension.");
    MatrixIndexT I = A.NumRows(), J = A.NumCols(), K = B.NumCols();
    const double *Aptr = A.Data(), *Bptr = B.Data();
    double *dest_ptr = dest.Data();
    MatrixIndexT aStride = A.Stride(),
                 bStride = B.Stride(),
                 destStride = dest.Stride();
    for (MatrixIndexT i = 0; i < I; i++) {
        for (MatrixIndexT k = 0; k < K; k++) {
            for (MatrixIndexT j = 0; j < J; j++) {
                dest_ptr[i*destStride+k] += Aptr[i*aStride+j] * Bptr[j*bStride+k];
            }
        }
    }
}

void Psvm::init(bool isRandom, int32 dim) {
    KALDI_ASSERT(dim > 1);
    Lambda_.Resize(dim, kUndefined);
    Gamma_.Resize(dim, kUndefined);
    c_.Resize(dim, kUndefined);
    if (isRandom) {
        RandFullCova(&Lambda_);
        KALDI_LOG << "1";
        RandFullCova(&Gamma_);
        Lambda_.Scale(-1.0);
        Gamma_.Scale(1.0);
    } else {
        Lambda_.SetZero();
        Lambda_.SetDiag(-1.0);
        Gamma_.SetZero();
        Gamma_.SetDiag(1.0);
    }
    c_.SetRandn();
    k_ = RandUniform();
}

void Psvm::precompute(const Matrix<double> *pivecs) {
    int32 dimIvec = pivecs->NumCols();
    int32 numIvec = pivecs->NumRows();
    KALDI_ASSERT(Dim()==dimIvec); // model paras should be initi at first
    Timer timer1;
    if (NumIvec() != numIvec) {
        // resize Lambda_x;
        Lambda_x_.Resize(numIvec, dimIvec, kUndefined);
        xt_Gamma_x_.Resize(numIvec, kUndefined);
        xt_c_.Resize(numIvec, kUndefined);
    }
    KALDI_LOG << "Inside of precompute. " << timer1.Elapsed()  << " seconds elapsed for Resize()";
    Timer timer2;
    Lambda_x_.AddMatSp(1.0, *pivecs, kNoTrans, Lambda_, 0.0);
    KALDI_LOG << "Inside of precompute. " << timer2.Elapsed() << " seconds elapsed for AddMatSp()";
    // for (size_t i=0; i < numIvec; ++i) {
    //     xt_Gamma_x_(i) = VecSpVec(pivecs->Row(i), Gamma_, pivecs->Row(i));
    // }
    Timer timer3;
    static Matrix<double> Gamma_x(numIvec, dimIvec, kUndefined);
    Gamma_x.AddMatSp(1.0, *pivecs, kNoTrans, Gamma_, 0.0);
    KALDI_LOG << "Inside of precompute. " << timer3.Elapsed()  << " seconds elpased for xt_c_ computation";
    xt_Gamma_x_.AddDiagMatMat(1.0, Gamma_x, kNoTrans, *pivecs, kTrans, 0.0);
    KALDI_LOG << "Inside of precompute. " << timer3.Elapsed()  << " seconds elpased for xt_c_ computation";

    xt_c_.AddMatVec(1.0, *pivecs, kNoTrans, c_, 0.0);
    KALDI_LOG << "Inside of precompute. " << timer3.Elapsed()  << " seconds elpased for xt_c_ computation";
    px_ = pivecs;
}

float Psvm::scoring(const int32 idx1, const int32 idx2) const {
    static double score = 0.0;
    score = 2 * VecVec(Lambda_x_.Row(idx1), px_->Row(idx2));
    score += xt_Gamma_x_(idx1) + xt_Gamma_x_(idx2);
    score += xt_c_(idx1) + xt_c_(idx2);
    score += k_;
    return score;
}

void Psvm::Write(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<Lambda>");
    Lambda_.Write(os, binary);
    WriteToken(os, binary, "<Gamma>");
    Gamma_.Write(os, binary);
    WriteToken(os, binary, "<c>");
    c_.Write(os, binary);
    WriteToken(os, binary, "<k>");
    WriteBasicType(os, binary, k_);
}

void Psvm::Read(std::istream &is, bool binary) {
    ExpectToken(is, binary, "<Lambda>");
    Lambda_.Read(is, binary);
    ExpectToken(is, binary, "<Gamma>");
    Gamma_.Read(is, binary);
    ExpectToken(is, binary, "<c>");
    c_.Read(is, binary);
    ExpectToken(is, binary, "<k>");
    ReadBasicType(is, binary, &k_);
}

PsvmEstimator::PsvmEstimator(bool random_init,
                             const Matrix<double> *pivecs):
    pivecs_(pivecs) {
    ppsvm_->init(random_init, DimIvec());
    // ResetStats();
}

// Retrain a existing PSVM
PsvmEstimator::PsvmEstimator(Psvm *ppsvm,
                             const Matrix<double> *pivecs):
    pivecs_(pivecs), ppsvm_(ppsvm) {
    KALDI_ASSERT(ppsvm_ != NULL);
    // ResetStats();    
}




void PsvmEstimator::ResetStats() {
    int32 num = NumIvec();
    int32 dim = DimIvec();
    KALDI_ASSERT(num > 0 && dim > 0);
    stats_1th_.Resize(num, dim);
    stats_0th_row_.Resize(num);
    stats_0th_col_.Resize(num);
    clf_loss_ = 0.0;
    sv_num_ = 0;
}

void PsvmEstimator::Forward(const std::vector<PsvmPair> &psvmPairVec, const size_t start_idx, const size_t end_idx, float pos_weight) {
    KALDI_ASSERT(start_idx >= 0 && end_idx <= psvmPairVec.size());
    // int32 num = NumIvec();
    // int32 dim = DimIvec();
    Timer timer1;
    ppsvm_->precompute(pivecs_);
    KALDI_LOG << timer1.Elapsed() << " seconds elapsed for finishing precompute";
    Timer timer2;
    ResetStats();
    KALDI_LOG << timer2.Elapsed() << " seconds elapsed for finishing ResetStats()";
    Timer timer3;
    // accumulate stats
    float score;
    for (size_t idx=start_idx; idx<end_idx; ++idx) {
        const PsvmPair &pair = psvmPairVec[idx];
        int32 i = pair.i;
        int32 j = pair.j;
        float label = pair.label;
        score = ppsvm_->scoring(i, j);
        if (score * label < 1.0f) {
            float weight_label = -1.0f/(pos_weight+1.0f);
            if (1.0f == label) {
                weight_label += 1.0f;
            }
            stats_1th_.Row(i).AddVec(weight_label, pivecs_->Row(j));
            stats_0th_row_(i) += weight_label;
            stats_0th_col_(j) += weight_label;
            clf_loss_ += (1-score*label) * abs(weight_label);
            sv_num_++;
        }
    }
    clf_loss_ = clf_loss_ / (double)(end_idx - start_idx);
    KALDI_LOG << timer3.Elapsed() << " seconds elapsed for finishing AccumulateStats";
}

void PsvmEstimator::Backward(double lr, double update_weight) const {
    int32 num = NumIvec();
    int32 dim = DimIvec();
    Matrix<double> temp_Lambda(dim, dim, kUndefined);
    SpMatrix<double> delta_Lambda(dim, kUndefined);
    SpMatrix<double> delta_Gamma(dim, kUndefined);
    Vector<double> sv_ys(num, kUndefined);
    Vector<double> delta_c(dim, kUndefined);
    double delta_k;

    temp_Lambda.AddMatMat(2.0, stats_1th_, kTrans, *pivecs_, kNoTrans, 0.0);
    delta_Lambda.CopyFromMat(temp_Lambda, kTakeMean);
    sv_ys.CopyFromVec(stats_0th_row_);
    sv_ys.AddVec(1.0, stats_0th_col_);
    delta_Gamma.AddMat2Vec(1.0, *pivecs_, kTrans, sv_ys, 0.0);
    delta_c.AddMatVec(1.0, *pivecs_, kTrans, sv_ys, 0.0);
    delta_k = stats_0th_col_.Sum();

    // update paras
    ppsvm_->Lambda_.Scale(1.0-lr);
    ppsvm_->Lambda_.AddSp(lr * update_weight, delta_Lambda);
    ppsvm_->Gamma_.Scale(1.0-lr);
    ppsvm_->Gamma_.AddSp(lr * update_weight, delta_Gamma);
    ppsvm_->c_.Scale(1.0-lr);
    ppsvm_->c_.AddVec(lr * update_weight, delta_c);
    ppsvm_->k_ = ppsvm_->k_ * (1.0-lr) + lr * update_weight * delta_k;
}

double PsvmEstimator::GetClfLoss() const {
    return clf_loss_;
}

double PsvmEstimator::GetParaLoss() const {
    double para_norm = 0.0;
    para_norm += ppsvm_->Lambda_.FrobeniusNorm();
    para_norm += ppsvm_->Gamma_.FrobeniusNorm();
    para_norm += ppsvm_->c_.Norm(2.0);
    para_norm += abs(ppsvm_->k_);
    return para_norm;
}

void PsvmEstimator::OneIter(const std::vector<PsvmPair> &psvmPairVec,
                              const size_t start_idx,
                              const size_t end_idx,
                              double neg_pos_ratio,
                              double lr,
                              double penalty) {
    double batch_size = end_idx - start_idx;
    double update_weight = penalty / batch_size;
    Timer timer;
    Forward(psvmPairVec, start_idx, end_idx, neg_pos_ratio);
    BaseFloat time = timer.Elapsed();
    KALDI_LOG << time << " seconds elapsed for finishing forward";

    KALDI_LOG << "Classification loss is " << GetClfLoss();
    KALDI_LOG << "Parameter Norm is " << GetParaLoss();
    KALDI_LOG << "Number of Support Vectors(SVs) is " << sv_num_;
    KALDI_LOG << "Sparsity of SVs is " << double(sv_num_) / batch_size * 100.0 << "%";
    Timer timer_backward;
    Backward(lr, update_weight);
    KALDI_LOG << timer.Elapsed() << " seconds elapsed for finishing backward";
}


} // namespace kaldi
