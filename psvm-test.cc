#include "psvm/psvm.cc"
#include "base/timer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"

int main(int argc, char *argv[]) {
    using namespace kaldi;
    try {

        Matrix<double> dest(140000, 500, kSetZero);
        Matrix<double> A(140000, 500, kSetZero);
        Matrix<double> B(500, 500, kSetZero);
        for (MatrixIndexT i = 0; i < 500; i++) {
            A(i, i) = 1;
            B(i, i) = 1;
        }
        //Timer timer1;
        //EfficientMatMul(dest, A, B);
        //KALDI_LOG << timer1.Elapsed() << " seconds eclapsed for EfficientMatMul";
        Timer timer2;
        dest.AddMatMat(1.0, A, kNoTrans, B, kNoTrans, 0.0);
        KALDI_LOG << timer2.Elapsed() << " seconds eclapsed for AddMatMat()";
        
        return 0;
    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}

