// psvmbin/psvm-generate-pairs.cc
//
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
//

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "psvm/psvm.h"

namespace kaldi {

struct PsvmPairConfig {
    float neg_pos_ratio;
    PsvmPairConfig(): neg_pos_ratio(10.0) { }
    void Register(OptionsItf *opts) {
        opts->Register("neg-pos-ratio", &neg_pos_ratio,
                "The ratio of #pairs belong to different speakers over #pairs "
                "belong to the same speaker.");
    }
};

}

int main(int argc, char *argv[]) {
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    typedef std::string string;
    using namespace kaldi;

    try {
        const char *usage=
            "It generates speaker-vector pairs for the training of PSVM.\n"
            "The number of pairs may be tremendously large without pre-selection,\n"
            "for 10M utterance, there are about 50G pairs.\n"
            "This program implements the random selection method proposed by Sandro.\n"
            "For the details, please refer to \n"
            "Large-Scale Training of Pairwise Support Vector Machines for Speaker Recognition.\n"
            "Each row of the output is : <utt1-idx> <utt2-idx> <label>\n"
            "<utt-idx> is the index in the <ivectro-rspecifier> file, <label> is 1 or -1."
            "Usage: psvm-generate-pairs <spk2utt-rspecifier> <ivector-rspecifier> <pairs-rxfilename>\n"
            "\n"
            "e.g.: psvm-generate-pairs ark:data/train/spk2utt scp:exp/ivectors_train/ivector.scp\n"
            " exp/psvm/train_pairs\n";

        ParseOptions po(usage);

        bool binary = true;
        PsvmPairConfig pair_config;


        po.Register("binary", &binary, "Write output in binary mode. By default, true");
        pair_config.Register(&po);

        po.Read(argc, argv);

        if (po.NumArgs() != 3) {
            po.PrintUsage();
            exit(1);
        }

        if (pair_config.neg_pos_ratio < 0.0) {
            KALDI_ERR << "Invalid argument: neg-pos-ratio should >= 0.0";
        }
        // Timer timer;
        string spk2utt_rspecifier = po.GetArg(1),
            ivector_rspecifier = po.GetArg(2),
            pair_wxfilename = po.GetArg(3);

        SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
        SequentialBaseFloatVectorReader ivector_reader(ivector_rspecifier);
        Output psvmPairWriter(pair_wxfilename, binary);
        std::vector<PsvmPair> pairVec;

        typedef unordered_map<string, size_t, StringHasher> HashType;
        HashType utt2idx;

        KALDI_LOG << "Reading utterance embeddings";
        for (size_t i=0; !ivector_reader.Done(); ivector_reader.Next(), i++) {
            utt2idx[ivector_reader.Key()] = i;
        }

        KALDI_LOG << "Reading spk2utt";
        unordered_map<string, std::vector<string>, StringHasher> spk2utt; 
        std::vector<string> spks;
        size_t num_pairs_pos = 0;
        size_t num_pairs_neg = 0;
        for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
            const string &spk = spk2utt_reader.Key();
            const std::vector<string> &utts = spk2utt_reader.Value();
            spk2utt[spk] = utts;
            spks.push_back(spk);
            // genereate pairs that belong to the same speaker
            size_t num_utts = utts.size();
            if (num_utts < 2) {
                KALDI_WARN << "Skip speaker " << spk << " with only " << num_utts
                    << " utterances.";
                continue;
            }
            for (size_t i=0; i < num_utts-1; ++i) {
                size_t utti = utt2idx[utts[i]];
                for (size_t j=i+1; j < num_utts; ++j) {
                    pairVec.push_back(PsvmPair(utti, utt2idx[utts[j]], 1.0f));
                    // ko.Stream() << utti << ' ' << utt2idx[utts[j]]
                    // vocab<< ' ' << 1.0 << std::endl;
                    num_pairs_pos++;
                }
            }
        }
        size_t num_spks = spks.size();
        KALDI_LOG << "Generated " << num_pairs_pos << " postive pairs(the same speaker)"
            << " from " << num_spks << " speakers.";
        if (num_spks < 2)
            KALDI_ERR << "Less than two speakers! Failed to generate negative pairs.";

        int num_avg_neg = num_pairs_pos * pair_config.neg_pos_ratio * 2 
            / (num_spks * (num_spks-1)) + 1;
        // KALDI_LOG << "pass1";
        for (size_t spk_i=0; spk_i < num_spks-1; ++spk_i) {
            for (size_t spk_j=spk_i+1; spk_j < num_spks; ++spk_j) {
                // KALDI_LOG << "pass2";
                const std::vector<string> &utts_i = spk2utt[spks[spk_i]];
                // KALDI_LOG << "pass3";
                const std::vector<string> &utts_j = spk2utt[spks[spk_j]];
                int utts_i_num = utts_i.size()-1;
                int utts_j_num = utts_j.size()-1;
                for (size_t i=0; i < num_avg_neg; ++i) {
                    // KALDI_LOG << "pass4";
                    const string &utti = utts_i[RandInt(0, utts_i_num)];
                    // KALDI_LOG << "pass5";
                    const string &uttj = utts_j[RandInt(0, utts_j_num)];
                    // KALDI_LOG << "pass6";
                    pairVec.push_back(PsvmPair(utt2idx[utti], utt2idx[uttj], -1.0f));
                    // ko.Stream() << utt2idx[utti] << ' ' << utt2idx[uttj] << ' ' 
                    //    << -1.0 << std::endl;
                    num_pairs_neg++;
                }
            }
        }
        // Note that the output pairs have not been randomnized yet!
        KALDI_LOG << "Generated " << num_pairs_neg << 
                " negative pairs(the different speakers), " << "with " << num_pairs_neg
                + num_pairs_pos << "in total.";
        KALDI_LOG << "The pair order has not been randomnized.";
        psvmPairVecWrite(psvmPairWriter.Stream(), binary, pairVec);
        KALDI_LOG << "Pairs are saved into successfully." << pair_rxfilename;
        // double time = timer.Elapsed();
        // KALDI_LOG << ""
        return (num_pairs_pos + num_pairs_neg != 0 ? 0 : 1);
    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}

