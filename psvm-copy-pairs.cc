// psvmbin/psvm-copy-pairs.cc
//
// Copyright 2019 Jerry Peng
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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "psvm/psvm.h"

int main(int argc, char *argv[]) {
    using namespace kaldi;
    try {
        const char *usage =
            "Copy PSVM pairs\n"
            "Usage: psvm-copy-pairs [options] <pairs-in> <pairs-out>\n"
            "e.g.:\n"
            "psvm-copy-pairs --binary=false exp/psvm/pairs.data -| less\n";
        bool binary = true;
        ParseOptions po(usage);
        po.Register("binary", &binary, "Write output in binary format");

        po.Read(argc, argv);

        if (po.NumArgs() != 2) {
            po.PrintUsage();
            exit(1);
        }

        std::string pair_rxfilename = po.GetArg(1),
            pair_wxfilename = po.GetArg(2);

        std::vector<PsvmPair> pairVec;
        bool binary_in;
        Input psvmPairReader(pair_rxfilename, &binary_in);
        Output psvmPairWriter(pair_wxfilename, binary);
        KALDI_LOG << "Read PSVM pairs from " << pair_rxfilename;
        psvmPairVecRead(psvmPairReader.Stream(), binary_in, pairVec);
        KALDI_LOG << "Write PSVM pairs to " << pair_wxfilename;
        psvmPairVecWrite(psvmPairWriter.Stream(), binary, pairVec);
        return 0;
    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}
