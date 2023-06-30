/*!
 * Copyright 2014-2022 by XGBoost Contributors
 * Copyright 2018 by Hongge Chen, Huan Zhang (robust split finding, RobustTrees)
 * \file updater_robust_colmaker.cc
 * \brief Column-wise exact tree maker with robust split finding, from
 *        "Robust Decision Trees Against Adversarial Examples" (Chen et al., ICML 2019).
 *        Port of src/tree/updater_robust_colmaker.cc of
 *        https://github.com/chenhongge/RobustTrees (XGBoost 0.72) onto the
 *        XGBoost 1.7 updater_colmaker.cc infrastructure.
 * \author Tianqi Chen, Hongge Chen, Huan Zhang
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <iostream>
#include <map>
#include <set>
#include <vector>

#include "xgboost/parameter.h"
#include "xgboost/tree_updater.h"
#include "xgboost/logging.h"
#include "xgboost/json.h"
#include "param.h"
#include "constraints.h"
#include "../common/random.h"
#include "split_evaluator.h"

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_robust_colmaker);

struct RobustColMakerTrainParam : XGBoostParameter<RobustColMakerTrainParam> {
  // speed optimization for dense column
  float opt_dense_col;
  // default direction choice
  int default_direction;
  // robust training eps for maximizing split loss
  float robust_eps;
  // verbose parameter for robust training
  bool robust_training_verbose;

  DMLC_DECLARE_PARAMETER(RobustColMakerTrainParam) {
    DMLC_DECLARE_FIELD(opt_dense_col)
        .set_range(0.0f, 1.0f)
        .set_default(1.0f)
        .describe("EXP Param: speed optimization for dense column.");
    DMLC_DECLARE_FIELD(default_direction)
        .set_default(0)
        .add_enum("learn", 0)
        .add_enum("left", 1)
        .add_enum("right", 2)
        .describe("Default direction choice when encountering a missing value");
    // Same names and defaults as RobustTrees src/tree/param.h.
    DMLC_DECLARE_FIELD(robust_eps)
        .set_lower_bound(0.0f)
        .set_default(0.3f)
        .describe("When robust training is used, define the epsilon range to maximize split loss");
    DMLC_DECLARE_FIELD(robust_training_verbose)
        .set_default(false)
        .describe("print information for robust training debugging");
  }

  /*! \brief whether need forward small to big search: default right */
  inline bool NeedForwardSearch(float col_density, bool indicator) const {
    return default_direction == 2 ||
           (default_direction == 0 && (col_density < opt_dense_col) && !indicator);
  }
  /*! \brief whether need backward big to small search: default left */
  inline bool NeedBackwardSearch() const { return default_direction != 2; }
};

DMLC_REGISTER_PARAMETER(RobustColMakerTrainParam);

/*! \brief column-wise update to construct a robust tree */
class RobustColMaker: public TreeUpdater {
 public:
  explicit RobustColMaker(GenericParameter const *ctx) : TreeUpdater(ctx) {}
  void Configure(const Args &args) override {
    param_.UpdateAllowUnknown(args);
    colmaker_param_.UpdateAllowUnknown(args);
  }

  void LoadConfig(Json const& in) override {
    auto const& config = get<Object const>(in);
    FromJson(config.at("train_param"), &this->param_);
    FromJson(config.at("robust_colmaker_train_param"), &this->colmaker_param_);
  }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["train_param"] = ToJson(param_);
    out["robust_colmaker_train_param"] = ToJson(colmaker_param_);
  }

  char const* Name() const override {
    return "robust_grow_colmaker";
  }

  void LazyGetColumnDensity(DMatrix *dmat) {
    // Finds densities if we don't already have them
    if (column_densities_.empty()) {
      std::vector<size_t> column_size(dmat->Info().num_col_);
      for (const auto &batch : dmat->GetBatches<SortedCSCPage>()) {
        auto page = batch.GetView();
        for (auto i = 0u; i < batch.Size(); i++) {
          column_size[i] += page[i].size();
        }
      }
      column_densities_.resize(column_size.size());
      for (auto i = 0u; i < column_densities_.size(); i++) {
        size_t nmiss = dmat->Info().num_row_ - column_size[i];
        column_densities_[i] =
            1.0f - (static_cast<float>(nmiss)) / dmat->Info().num_row_;
      }
    }
  }

  void Update(HostDeviceVector<GradientPair> *gpair, DMatrix *dmat,
              common::Span<HostDeviceVector<bst_node_t>> /*out_position*/,
              const std::vector<RegTree *> &trees) override {
    if (collective::IsDistributed()) {
      LOG(FATAL) << "Updater `robust_grow_colmaker` or `robust_exact` tree method doesn't "
                    "support distributed training.";
    }
    if (!dmat->SingleColBlock()) {
      LOG(FATAL) << "Updater `robust_grow_colmaker` or `robust_exact` tree method doesn't "
                    "support external memory training.";
    }
    this->LazyGetColumnDensity(dmat);
    // rescale learning rate according to size of trees
    float lr = param_.learning_rate;
    param_.learning_rate = lr / trees.size();
    interaction_constraints_.Configure(param_, dmat->Info().num_row_);
    // build tree
    for (auto tree : trees) {
      CHECK(ctx_);
      Builder builder(param_, colmaker_param_, interaction_constraints_, ctx_,
                      column_densities_);
      builder.Update(gpair->ConstHostVector(), dmat, tree);
    }
    param_.learning_rate = lr;
  }

 protected:
  // training parameter
  TrainParam param_;
  RobustColMakerTrainParam colmaker_param_;
  std::vector<float> column_densities_;

  FeatureInteractionConstraintHost interaction_constraints_;
  // data structure
  /*! \brief per thread x per node entry to store tmp data */
  struct ThreadEntry {
    /*! \brief statistics of data */
    GradStats stats;
    /*! \brief statistics of data: <eta */
    GradStats stats_left;
    /*! \brief statistics of data: <eta-eps */
    GradStats stats_c_left;
    unsigned int c_left_counter { 0 };
    /*! \brief statistics of data: [eta, eta+eps) should be the stats of data_unc_right */
    GradStats stats_unc_right;
    /*! \brief statistics of data: [eta-eps, eta+eps) should be the stats of data_unc */
    GradStats stats_unc;
    /*! \brief queue of seen data not added to stats_left: [eta, eta+eps) */
    std::deque<const Entry*> data_unc_right;
    /*! \brief queue of uncertain data: [eta-eps, eta+eps) */
    std::deque<const Entry*> data_unc;
    /*! \brief last feature value scanned */
    bst_float last_fvalue { 0 };
    /*! \brief current best solution */
    SplitEntry best;
    // constructor
    ThreadEntry() = default;
  };
  struct NodeEntry {
    /*! \brief statics for node entry */
    GradStats stats;
    /*! \brief loss of this node, without split */
    bst_float root_gain { 0.0f };
    /*! \brief weight calculated related to current data */
    bst_float weight { 0.0f };
    /*! \brief current best solution */
    SplitEntry best;
    // constructor
    NodeEntry() = default;
  };
  // actual builder that runs the algorithm
  class Builder {
   public:
    // constructor
    explicit Builder(const TrainParam &param, const RobustColMakerTrainParam &colmaker_train_param,
                     FeatureInteractionConstraintHost _interaction_constraints,
                     GenericParameter const *ctx, const std::vector<float> &column_densities)
        : param_(param),
          score_param_(param),
          colmaker_train_param_{colmaker_train_param},
          ctx_{ctx},
          tree_evaluator_(param_, column_densities.size(), GenericParameter::kCpuId),
          interaction_constraints_{std::move(_interaction_constraints)},
          column_densities_(column_densities) {
      // The 0.72 SplitEvaluator (elastic_net) used by RobustTrees never zeroed the
      // weight/gain of a child below min_child_weight; it only applied min_child_weight
      // in the explicit checks of the split search.  The robust search scores
      // partitions (uncertain points all left/right, swapped) whose sides are not
      // checked, so the 1.7 evaluator is fed a copy with min_child_weight = 0 to keep
      // RobustTrees scores and leaf weights.  The explicit checks still use param_.
      score_param_.min_child_weight = 0.0f;
    }
    // update one tree, growing
    virtual void Update(const std::vector<GradientPair>& gpair,
                        DMatrix* p_fmat,
                        RegTree* p_tree) {
      std::vector<int> newnodes;
      this->InitData(gpair, *p_fmat);
      this->InitNewNode(qexpand_, gpair, *p_fmat, *p_tree);
      // We can check max_leaves too, but might break some grid searching pipelines.
      CHECK_GT(param_.max_depth, 0) << "robust_exact tree method doesn't support unlimited depth.";
      for (int depth = 0; depth < param_.max_depth; ++depth) {
        this->FindSplit(depth, qexpand_, gpair, p_fmat, p_tree);
        this->ResetPosition(qexpand_, p_fmat, *p_tree);
        this->UpdateQueueExpand(*p_tree, qexpand_, &newnodes);
        this->InitNewNode(newnodes, gpair, *p_fmat, *p_tree);
        for (auto nid : qexpand_) {
          if ((*p_tree)[nid].IsLeaf()) {
            continue;
          }
          int cleft = (*p_tree)[nid].LeftChild();
          int cright = (*p_tree)[nid].RightChild();

          tree_evaluator_.AddSplit(nid, cleft, cright, snode_[nid].best.SplitIndex(),
                                   snode_[cleft].weight, snode_[cright].weight);
          interaction_constraints_.Split(nid, snode_[nid].best.SplitIndex(), cleft, cright);
        }
        qexpand_ = newnodes;
        // if nothing left to be expand, break
        if (qexpand_.size() == 0) break;
      }
      // set all the rest expanding nodes to leaf
      for (const int nid : qexpand_) {
        (*p_tree)[nid].SetLeaf(snode_[nid].weight * param_.learning_rate);
      }
      // remember auxiliary statistics in the tree node
      for (int nid = 0; nid < p_tree->param.num_nodes; ++nid) {
        p_tree->Stat(nid).loss_chg = snode_[nid].best.loss_chg;
        p_tree->Stat(nid).base_weight = snode_[nid].weight;
        p_tree->Stat(nid).sum_hess = static_cast<float>(snode_[nid].stats.sum_hess);
      }
    }

   protected:
    // initialize temp data structure
    inline void InitData(const std::vector<GradientPair>& gpair,
                         const DMatrix& fmat) {
      {
        // setup position
        position_.resize(gpair.size());
        CHECK_EQ(fmat.Info().num_row_, position_.size());
        std::fill(position_.begin(), position_.end(), 0);
        // mark delete for the deleted datas
        for (size_t ridx = 0; ridx < position_.size(); ++ridx) {
          if (gpair[ridx].GetHess() < 0.0f) position_[ridx] = ~position_[ridx];
        }
        // mark subsample
        if (param_.subsample < 1.0f) {
          CHECK_EQ(param_.sampling_method, TrainParam::kUniform)
            << "Only uniform sampling is supported, "
            << "gradient-based sampling is only support by GPU Hist.";
          std::bernoulli_distribution coin_flip(param_.subsample);
          auto& rnd = common::GlobalRandom();
          for (size_t ridx = 0; ridx < position_.size(); ++ridx) {
            if (gpair[ridx].GetHess() < 0.0f) continue;
            if (!coin_flip(rnd)) position_[ridx] = ~position_[ridx];
          }
        }
      }
      {
        column_sampler_.Init(fmat.Info().num_col_, fmat.Info().feature_weights.ConstHostVector(),
                             param_.colsample_bynode, param_.colsample_bylevel,
                             param_.colsample_bytree);
      }
      {
        // setup temp space for each thread
        // reserve a small space
        stemp_.clear();
        stemp_.resize(this->ctx_->Threads(), std::vector<ThreadEntry>());
        for (auto& i : stemp_) {
          i.clear(); i.reserve(256);
        }
        snode_.reserve(256);
      }
      {
        // expand query
        qexpand_.reserve(256); qexpand_.clear();
        qexpand_.push_back(0);
      }
    }
    /*!
     * \brief initialize the base_weight, root_gain,
     *  and NodeEntry for all the new nodes in qexpand
     */
    inline void InitNewNode(const std::vector<int>& qexpand,
                            const std::vector<GradientPair>& gpair,
                            const DMatrix& fmat,
                            const RegTree& tree) {
      {
        // setup statistics space for each tree node
        for (auto& i : stemp_) {
          i.resize(tree.param.num_nodes, ThreadEntry());
        }
        snode_.resize(tree.param.num_nodes, NodeEntry());
      }
      const MetaInfo& info = fmat.Info();
      // setup position
      common::ParallelFor(info.num_row_, ctx_->Threads(), [&](auto ridx) {
        int32_t const tid = omp_get_thread_num();
        if (position_[ridx] < 0) return;
        stemp_[tid][position_[ridx]].stats.Add(gpair[ridx]);
      });
      // sum the per thread statistics together
      for (int nid : qexpand) {
        GradStats stats;
        for (auto& s : stemp_) {
          stats.Add(s[nid].stats);
        }
        // update node statistics
        snode_[nid].stats = stats;
      }

      auto evaluator = tree_evaluator_.GetEvaluator();
      // calculating the weights
      for (int nid : qexpand) {
        bst_node_t parentid = tree[nid].Parent();
        snode_[nid].weight = static_cast<float>(
            evaluator.CalcWeight(parentid, score_param_, snode_[nid].stats));
        snode_[nid].root_gain = static_cast<float>(
            evaluator.CalcGain(parentid, score_param_, snode_[nid].stats));
      }
    }
    /*! \brief update queue expand add in new leaves */
    inline void UpdateQueueExpand(const RegTree& tree,
                                  const std::vector<int> &qexpand,
                                  std::vector<int>* p_newnodes) {
      p_newnodes->clear();
      for (int nid : qexpand) {
        if (!tree[ nid ].IsLeaf()) {
          p_newnodes->push_back(tree[nid].LeftChild());
          p_newnodes->push_back(tree[nid].RightChild());
        }
      }
    }

    static inline void SubtractGpair(GradStats *stats, GradientPair const &p) {
      stats->Add(-p.GetGrad(), -p.GetHess());
    }
    static inline GradStats Union(GradStats a, GradStats const &b) {
      a.Add(b);
      return a;
    }

    // robust enumeration of the split values of a specific feature (RobustTrees)
    void EnumerateSplit(
        const Entry *begin, const Entry *end, int d_step, bst_uint fid,
        const std::vector<GradientPair> &gpair,
        std::vector<ThreadEntry> &temp, // NOLINT(*)
        TreeEvaluator::SplitEvaluator<TrainParam> const &evaluator) const {
      const bool verbose = colmaker_train_param_.robust_training_verbose;
      // RobustTrees score of a split candidate, ComputeSplitScore(nid, fid, l, r) - root_gain.
      // RobustTrees passes (right, left) on the backward pass (d_step == -1); kept as is.
      auto loss_change = [&](int nid, GradStats const &left, GradStats const &right) {
        float gain = d_step == -1
                         ? evaluator.CalcSplitGain(score_param_, nid, fid, right, left)
                         : evaluator.CalcSplitGain(score_param_, nid, fid, left, right);
        return static_cast<bst_float>(gain - snode_[nid].root_gain);
      };

      // check descent ordering or ascent ordering.
      bool descent = (begin->fvalue) > ((end - d_step)->fvalue);

      if (verbose) {
        printf("\n\n\n########################## We are now using the new robust xgboost "
               "code on feature id %d ##########################\n", static_cast<int>(fid));
        printf("end-begin=%ld\n", static_cast<long>(end - begin));  // NOLINT(*)
        printf("d_step is %d\n", d_step);
      }

      // here we enforce ascent order since in xgboost the split is defined as <eta vs. >=eta
      const Entry *first;
      if (descent) {
        if (verbose) {
          printf("\n descent ordering. begin = %4.5f, end = %4.5f, need to use <=eta vs. >eta. "
                 "swap the order!\n", begin->fvalue, (end - d_step)->fvalue);
        }
        first = end - d_step;
      } else {
        if (verbose) {
          printf("\n ascent ordering. begin = %4.5f, end = %4.5f, need to use <eta vs. >=eta  \n",
                 begin->fvalue, (end - d_step)->fvalue);
        }
        first = begin;
      }
      const int it_step = descent ? -d_step : d_step;

      const int length = std::abs(begin - end);
      int counter = 1;

      if (verbose) {
        printf("\n feature values:\n");
        counter = 1;
        for (const Entry *it = first; counter <= length; it += it_step, counter++) {
          printf("%4.5f ", it->fvalue);
        }
        printf("\n");
      }

      const std::vector<int> &qexpand = qexpand_;
      // clear all the temp statistics
      for (auto nid : qexpand) {
        temp[nid].stats = GradStats();
        temp[nid].stats_left = GradStats();
        temp[nid].data_unc_right.clear();
        temp[nid].data_unc.clear();
        temp[nid].stats_unc_right = GradStats();
        temp[nid].stats_c_left = GradStats();
        temp[nid].c_left_counter = 0;
        temp[nid].stats_unc = GradStats();
      }

      const bst_float eps = static_cast<bst_float>(colmaker_train_param_.robust_eps);
      if (verbose) printf("epsilon is %4.5f\n", eps);
      // left statistics
      GradStats c;

      if (verbose) {
        printf("\n******************************* feature id %d main loop starts "
               "************************************", static_cast<int>(fid));
      }

      std::map<int, int> n_node_point;
      if (verbose) {  // if verbose, need to count points at each node
        counter = 1;
        for (const Entry *it = first; counter <= length; it += it_step, counter++) {
          n_node_point[position_[it->index]]++;
        }
      }
      counter = 1;
      for (const Entry *it = first; counter <= length; it += it_step, counter++) {
        const bst_uint ridx = it->index;
        const int nid = position_[ridx];

        if (verbose) {
          printf("\n------------------------------------------------------------------------\n");
          printf("nid: %d, it-begin %ld, fid %d\n", nid, static_cast<long>(it - begin),  // NOLINT(*)
                 static_cast<int>(fid));
        }

        // interaction constraints are a 1.x feature, checked as in grow_colmaker
        if (nid < 0 || !interaction_constraints_.Query(nid, fid)) { continue; }
        // start working
        const bst_float fvalue = it->fvalue;

        if (verbose) printf("it->fvalue: %4.5f\n", it->fvalue);

        // if we are using descent order, eta = x + eps, if we are using ascent order, eta = x - eps
        bst_float eta = fvalue - eps;

        if (verbose) {
          printf("eta %4.5f, uncertainty range: %4.5f ~ %4.5f, gradient:%4.5f, hessian:%4.5f\n",
                 eta, eta - eps, eta + eps, gpair[ridx].GetGrad(), gpair[ridx].GetHess());
        }

        // get the statistics of nid
        ThreadEntry &e = temp[nid];
        // test if first hit, this is fine, because we set 0 during init
        if (e.stats.Empty()) {
          if (verbose) printf("first hit\n");
          e.stats.Add(gpair[ridx]);
          e.last_fvalue = fvalue;
          e.data_unc_right.push_back(it);
          e.data_unc.push_back(it);
          e.stats_unc_right.Add(gpair[ridx]);
          e.stats_unc.Add(gpair[ridx]);
        } else {
          if (verbose) {
            printf("e.data_unc_right: %4.5f ~ %4.5f\n", e.data_unc_right.front()->fvalue,
                   e.data_unc_right.back()->fvalue);
            printf("e.data_unc: %4.5f ~ %4.5f\n", e.data_unc.front()->fvalue,
                   e.data_unc.back()->fvalue);
          }

          // add the unadded data to stats_left and remove them from the queue of data
          // that <eta but unadded
          while (!e.data_unc_right.empty()) {
            const Entry *unadded_front = e.data_unc_right.front();
            if (unadded_front->fvalue < eta) {
              const bst_uint unadded_front_ridx = unadded_front->index;
              e.stats_left.Add(gpair[unadded_front_ridx]);
              SubtractGpair(&e.stats_unc_right, gpair[unadded_front_ridx]);
              e.data_unc_right.pop_front();
            } else {
              break;
            }
          }
          // remove the data from queue of data that is in the uncertain range
          while (!e.data_unc.empty()) {
            const Entry *unc_front = e.data_unc.front();
            if (unc_front->fvalue < eta - eps) {
              e.data_unc.pop_front();
              const bst_uint unc_front_ridx = unc_front->index;
              e.stats_c_left.Add(gpair[unc_front_ridx]);
              e.c_left_counter++;
              SubtractGpair(&e.stats_unc, gpair[unc_front_ridx]);
            } else {
              break;
            }
          }
          // try to find a split
          if (verbose) {
            printf("fvalue != e.last_fvalue: %d, e.stats.sum_hess >= param_.min_child_weight: %d\n",
                   static_cast<int>(fvalue != e.last_fvalue),
                   static_cast<int>(e.stats.sum_hess >= param_.min_child_weight));
            printf("s.stats_left.sum_hess %4.5f \n", e.stats_left.sum_hess);
            // if c_right_counter <= 0, means we didn't count the number of points on this node
            unsigned int c_right_counter =
                n_node_point[nid] - e.c_left_counter - e.data_unc.size();
            printf("\n left certain:%u, uncertain:%zu, right certain:%u\n", e.c_left_counter,
                   e.data_unc.size(), c_right_counter);
          }

          if (fvalue != e.last_fvalue &&
              e.stats.sum_hess >= param_.min_child_weight) {
            c.SetSubstract(snode_[nid].stats, e.stats_left);
            if (verbose) {
              printf("c.sum_hess %4.5f, param_.min_child_weight %4.5f\n", c.sum_hess,
                     param_.min_child_weight);
            }
            if (c.sum_hess >= param_.min_child_weight) {
              if (verbose) printf("try to find a split\n");
              bst_float loss_chg = loss_change(nid, e.stats_left, c);
              if (verbose) printf("\ninitial loss change is %4.5f\n", loss_chg);

              // one-side/swap minimization
              if (e.data_unc.size() > 0) {
                if (verbose) printf("\n [start minimization]\n");
                // all uncertainty to left
                GradStats all_left = Union(e.stats_c_left, e.stats_unc);
                GradStats c_right;
                c_right.SetSubstract(snode_[nid].stats, all_left);
                bst_float put_left_loss_chg = loss_change(nid, all_left, c_right);
                if (verbose) printf("\n put all left loss change is %4.5f\n", put_left_loss_chg);
                if (put_left_loss_chg < loss_chg) {
                  loss_chg = put_left_loss_chg;
                  if (verbose) printf("\n update loss change to %4.5f\n", loss_chg);
                }
                // all uncertainty to right
                GradStats all_right;
                all_right.SetSubstract(snode_[nid].stats, e.stats_c_left);
                bst_float put_right_loss_chg = loss_change(nid, e.stats_c_left, all_right);
                if (verbose) printf("\n all right loss change is %4.5f\n", put_right_loss_chg);
                if (put_right_loss_chg < loss_chg) {
                  loss_chg = put_right_loss_chg;
                  if (verbose) printf("\n update loss change to %4.5f\n", loss_chg);
                }
                // swap
                GradStats swap_left = Union(e.stats_c_left, e.stats_unc_right);
                GradStats swap_right;
                swap_right.SetSubstract(snode_[nid].stats, swap_left);
                bst_float swap_loss_chg = loss_change(nid, swap_left, swap_right);
                if (verbose) printf("\n swapping loss change is %4.5f\n", swap_loss_chg);
                if (swap_loss_chg < loss_chg) {
                  loss_chg = swap_loss_chg;
                  if (verbose) printf("\n update loss change to %4.5f\n", loss_chg);
                }
              }
              // left_sum/right_sum are only kept by 1.x for ExpandNode bookkeeping; the
              // scan is ascending, so stats_left is the nominal left child here.
              e.best.Update(loss_chg, fid, eta, d_step == -1, false, e.stats_left, c);
            }
          }
          // update the statistics
          e.stats.Add(gpair[ridx]);
          e.last_fvalue = fvalue;
          // add data to the two queues
          e.data_unc_right.push_back(it);
          e.data_unc.push_back(it);
          e.stats_unc_right.Add(gpair[ridx]);
          e.stats_unc.Add(gpair[ridx]);
        }
        if (verbose) {
          printf("\n [main loop] current best split for nid %d is fid=%d, threshold=%4.5f and "
                 "with loss change=%4.5f", nid, static_cast<int>(e.best.SplitIndex()),
                 e.best.split_value, e.best.loss_chg);
        }
      }
      if (verbose) {
        printf("\n**************** We now finished the main loop on feature id %d, all "
               "statistics starts  ****************\n", static_cast<int>(fid));
      }
      // finish updating all statistics, check if it is possible to include all sum statistics
      for (int nid : qexpand) {
        ThreadEntry &e = temp[nid];
        c.SetSubstract(snode_[nid].stats, e.stats);
        if (e.stats.sum_hess >= param_.min_child_weight &&
            c.sum_hess >= param_.min_child_weight) {
          bst_float loss_chg = loss_change(nid, e.stats, c);
          if (verbose) printf("\nall sum statistics loss_chg: %4.5f \n", loss_chg);
          const bst_float gap = std::abs(e.last_fvalue) + kRtEps + eps;
          const bst_float delta = d_step == +1 ? gap : -gap;
          if (d_step == -1) {
            e.best.Update(loss_chg, fid, e.last_fvalue + delta, true, false, c, e.stats);
          } else {
            e.best.Update(loss_chg, fid, e.last_fvalue + delta, false, false, e.stats, c);
          }
        }
        if (verbose) {
          printf("\n [all statistics (nomid)] current best split for nid %d is fid=%d, "
                 "threshold=%4.5f and with loss change=%4.5f", nid,
                 static_cast<int>(e.best.SplitIndex()), e.best.split_value, e.best.loss_chg);
        }
      }

      // move thresholds to mid
      counter = 1;
      std::map<int, bst_float> last_fvalue_map;
      std::set<int> updated_nid;
      for (const Entry *it = first; counter <= length; it += it_step, counter++) {
        const bst_uint ridx = it->index;
        const int nid = position_[ridx];
        if (nid < 0) continue;
        ThreadEntry &e = temp[nid];
        if (e.best.SplitIndex() != fid || updated_nid.find(nid) != updated_nid.end()) {
          continue;
        }
        if (last_fvalue_map.find(nid) != last_fvalue_map.end()) {
          if (last_fvalue_map[nid] < e.best.split_value && e.best.split_value <= it->fvalue) {
            if (verbose) {
              std::cout << "\nmoved! " << "nodeid: " << nid << " fid: " << fid << '\t'
                        << last_fvalue_map[nid] << '\t' << e.best.split_value << '\t'
                        << it->fvalue << '\t'
                        << e.best.split_value - 0.5f * (it->fvalue + last_fvalue_map[nid])
                        << '\n';
            }
            e.best.split_value = (it->fvalue + last_fvalue_map[nid]) * 0.5f;
            if (verbose) std::cout << e.best.split_value << '\n';
            updated_nid.insert(nid);
          }
        }
        last_fvalue_map[nid] = it->fvalue;
      }
    }

    // update the solution candidate
    virtual void UpdateSolution(const SortedCSCPage &batch,
                                const std::vector<bst_feature_t> &feat_set,
                                const std::vector<GradientPair> &gpair, DMatrix *) {
      // start enumeration
      const auto num_features = feat_set.size();
      CHECK(this->ctx_);
      const int batch_size =  // NOLINT
          std::max(static_cast<int>(num_features / this->ctx_->Threads() / 32), 1);
      auto page = batch.GetView();
      common::ParallelFor(
          num_features, ctx_->Threads(), common::Sched::Dyn(batch_size), [&](auto i) {
            auto evaluator = tree_evaluator_.GetEvaluator();
            bst_feature_t const fid = feat_set[i];
            int32_t const tid = omp_get_thread_num();
            auto c = page[fid];
            // 0.72 dropped empty columns from the feature set; the robust scan
            // dereferences the first/last entry, so skip them here.
            if (c.size() == 0) { return; }
            const bool ind = c.size() != 0 && c[0].fvalue == c[c.size() - 1].fvalue;
            if (colmaker_train_param_.NeedForwardSearch(column_densities_[fid], ind)) {
              this->EnumerateSplit(c.data(), c.data() + c.size(), +1, fid, gpair, stemp_[tid],
                                   evaluator);
            }
            if (colmaker_train_param_.NeedBackwardSearch()) {
              this->EnumerateSplit(c.data() + c.size() - 1, c.data() - 1, -1, fid, gpair,
                                   stemp_[tid], evaluator);
            }
          });
    }
    // find splits at current level, do split per level
    inline void FindSplit(int depth,
                          const std::vector<int> &qexpand,
                          const std::vector<GradientPair> &gpair,
                          DMatrix *p_fmat,
                          RegTree *p_tree) {
      auto evaluator = tree_evaluator_.GetEvaluator();

      auto feat_set = column_sampler_.GetFeatureSet(depth);
      for (const auto &batch : p_fmat->GetBatches<SortedCSCPage>()) {
        this->UpdateSolution(batch, feat_set->HostVector(), gpair, p_fmat);
      }
      // after this each thread's stemp will get the best candidates, aggregate results
      this->SyncBestSolution(qexpand);
      // get the best result, we can synchronize the solution
      for (int nid : qexpand) {
        NodeEntry const &e = snode_[nid];
        // now we know the solution in snode[nid], set split
        if (e.best.loss_chg > kRtEps) {
          // child leaf values are provisional: InitNewNode recomputes them from the
          // actual partition and every child is either split or set as leaf later.
          bst_float left_leaf_weight =
              evaluator.CalcWeight(nid, score_param_, e.best.left_sum) *
              param_.learning_rate;
          bst_float right_leaf_weight =
              evaluator.CalcWeight(nid, score_param_, e.best.right_sum) *
              param_.learning_rate;
          p_tree->ExpandNode(nid, e.best.SplitIndex(), e.best.split_value,
                             e.best.DefaultLeft(), e.weight, left_leaf_weight,
                             right_leaf_weight, e.best.loss_chg,
                             e.stats.sum_hess,
                             e.best.left_sum.GetHess(), e.best.right_sum.GetHess(),
                             0);
        } else {
          (*p_tree)[nid].SetLeaf(e.weight * param_.learning_rate);
        }
      }
    }
    // reset position of each data points after split is created in the tree
    inline void ResetPosition(const std::vector<int> &qexpand,
                              DMatrix* p_fmat,
                              const RegTree& tree) {
      // set the positions in the nondefault
      this->SetNonDefaultPosition(qexpand, p_fmat, tree);
      // set rest of instances to default position
      // set default direct nodes to default
      // for leaf nodes that are not fresh, mark then to ~nid,
      // so that they are ignored in future statistics collection
      common::ParallelFor(p_fmat->Info().num_row_, this->ctx_->Threads(), [&](auto ridx) {
        CHECK_LT(ridx, position_.size()) << "ridx exceed bound "
                                         << "ridx=" << ridx << " pos=" << position_.size();
        const int nid = this->DecodePosition(ridx);
        if (tree[nid].IsLeaf()) {
          // mark finish when it is not a fresh leaf
          if (tree[nid].RightChild() == -1) {
            position_[ridx] = ~nid;
          }
        } else {
          // push to default branch
          if (tree[nid].DefaultLeft()) {
            this->SetEncodePosition(ridx, tree[nid].LeftChild());
          } else {
            this->SetEncodePosition(ridx, tree[nid].RightChild());
          }
        }
      });
    }
    // customization part
    // synchronize the best solution of each node
    virtual void SyncBestSolution(const std::vector<int> &qexpand) {
      for (int nid : qexpand) {
        NodeEntry &e = snode_[nid];
        CHECK(this->ctx_);
        for (int tid = 0; tid < this->ctx_->Threads(); ++tid) {
          e.best.Update(stemp_[tid][nid].best);
        }
      }
    }
    virtual void SetNonDefaultPosition(const std::vector<int> &qexpand,
                                       DMatrix *p_fmat,
                                       const RegTree &tree) {
      // step 1, classify the non-default data into right places
      std::vector<unsigned> fsplits;
      for (int nid : qexpand) {
        if (!tree[nid].IsLeaf()) {
          fsplits.push_back(tree[nid].SplitIndex());
        }
      }
      std::sort(fsplits.begin(), fsplits.end());
      fsplits.resize(std::unique(fsplits.begin(), fsplits.end()) - fsplits.begin());
      for (const auto &batch : p_fmat->GetBatches<SortedCSCPage>()) {
        auto page = batch.GetView();
        for (auto fid : fsplits) {
          auto col = page[fid];
          common::ParallelFor(col.size(), this->ctx_->Threads(), [&](auto j) {
            const bst_uint ridx = col[j].index;
            const int nid = this->DecodePosition(ridx);
            const bst_float fvalue = col[j].fvalue;
            // go back to parent, correct those who are not default
            if (!tree[nid].IsLeaf() && tree[nid].SplitIndex() == fid) {
              if (fvalue < tree[nid].SplitCond()) {
                this->SetEncodePosition(ridx, tree[nid].LeftChild());
              } else {
                this->SetEncodePosition(ridx, tree[nid].RightChild());
              }
            }
          });
        }
      }
    }
    // utils to get/set position, with encoded format
    // return decoded position
    inline int DecodePosition(bst_uint ridx) const {
      const int pid = position_[ridx];
      return pid < 0 ? ~pid : pid;
    }
    // encode the encoded position value for ridx
    inline void SetEncodePosition(bst_uint ridx, int nid) {
      if (position_[ridx] < 0) {
        position_[ridx] = ~nid;
      } else {
        position_[ridx] = nid;
      }
    }
    //  --data fields--
    const TrainParam& param_;
    // param_ with min_child_weight = 0, used for all evaluator scores and weights
    TrainParam score_param_;
    const RobustColMakerTrainParam& colmaker_train_param_;
    // number of omp thread used during training
    GenericParameter const* ctx_;
    common::ColumnSampler column_sampler_;
    // Instance Data: current node position in the tree of each instance
    std::vector<int> position_;
    // PerThread x PerTreeNode: statistics for per thread construction
    std::vector< std::vector<ThreadEntry> > stemp_;
    /*! \brief TreeNode Data: statistics for each constructed node */
    std::vector<NodeEntry> snode_;
    /*! \brief queue of nodes to be expanded */
    std::vector<int> qexpand_;
    TreeEvaluator tree_evaluator_;

    FeatureInteractionConstraintHost interaction_constraints_;
    const std::vector<float> &column_densities_;
  };
};

XGBOOST_REGISTER_TREE_UPDATER(RobustColMaker, "robust_grow_colmaker")
.describe("Grow tree with parallelization over columns, robust split finding (RobustTrees).")
.set_body([](GenericParameter const* ctx, ObjInfo) {
    return new RobustColMaker(ctx);
  });
}  // namespace tree
}  // namespace xgboost
