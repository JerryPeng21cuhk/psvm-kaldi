// psvmbin/psvm-init.cc

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
            "Initalize psvm classifier\n"
            "Usage: psvm-init [options] <psvm-out>\n"
            "e.g.:\n"
            " psvm-init psvm.mdl\n";

        bool binary = true;
        bool random_init = true;
        int32 ivec_dim = 600;
        ParseOptions po(usage);
        po.Register("ivec-dim", &ivec_dim, "The dimension of ivectors for the classification of Psvm.");
        po.Register("random-init", &random_init, "If true, randomly initialize PSVM parameters; Otherwise, set part of parameters to zero.");
        po.Register("binary", &binary, "Write output in binary mode");

        po.Read(argc, argv);

        if (po.NumArgs() != 1) {
            po.PrintUsage();
            exit(1);
        }

        std::string psvm_wxfilename = po.GetArg(1);
        // Output psvmWriter(psvm_wxfilename, binary);

        Psvm psvm;
        psvm.init(random_init, ivec_dim);
        // psvm.Write(psvmWriter.Stream(), binary);
        WriteKaldiObject(psvm, psvm_wxfilename, binary);
        KALDI_LOG << "Initialize PSVM with iVector dimension "
                  << ivec_dim << " and wrote it to "
                  << psvm_wxfilename;
        return 0;
    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}
