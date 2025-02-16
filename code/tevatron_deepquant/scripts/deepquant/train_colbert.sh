root=/mnt/nas/user/tevatron/outputs/deepquant
# path=$root/colbert_deepquant_fin2/test
# path=$root/colbert_deepquant_fin2/numgpt_relevance_qcolb_NOT_opinversion_relcross_pairsig/lossv2_alpha0.7_max_collate_attention_query_FFL_aggr_token
# path=$root/pairwiseprob_test/numgpt_rel/relraw_opPRED_numqcolb_pairprobCTX-sigmoid-MAX
path=$root/test_final_clsalpha_v2_clinical
# dataroot=/mnt/nas/user/QuantityAwareRankers/train_data/tevatron_files/new_withquants
dataroot=/mnt/nas/user/QuantityAwareRankers/train_data/clinical/final
data=$dataroot/tevatron_train
eval "$(conda shell.bash hook)"
conda activate /mnt/nas/user/condaEnvV3

mkdir -p $path
SOURCE=$0
TS=$(date +"%m-%d_%H-%M-%S")
DEST_PATH="$path/$(basename $SOURCE .sh)_$TS.sh"
cp $SOURCE $DEST_PATH
echo "Script copied successfully to $DEST_PATH"

# --num_as_token \
#   --numgpt_embed \
#   --embed_exp_frac 0.25 \
#   --num_pred_loss_alpha 1 \
#   --add_num_pred_loss \
  # --learning_rate 5e-6 \
CUDA_VISIBLE_DEVICES=0 python3 /mnt/nas/user/tevatron/src/tevatron/driver/train.py \
  --output_dir $path \
  --model_name_or_path colbert-ir/colbertv2.0 \
  --tokenizer_name bert-base-uncased \
  --save_strategy epoch \
  --dataset_name Tevatron/msmarco-passage \
  --train_dir $data \
  --dataset_format json \
  --bf16 \
  --per_device_train_batch_size 32 \
  --train_n_passages 8 \
  --q_max_len 32 \
  --q_len 32 \
  --p_max_len 128 \
  --p_len 128 \
  --num_train_epochs 8 \
  --logging_steps 500 \
  --overwrite_output_dir \
  --data_cache_dir /mnt/nas/user/.huggingfaceCachecolb_token_unit_token/ \
  --cache_dir /mnt/nas/user/.huggingfaceCachecolb_token_unit_token/ \
  --model colbert \
  --add_pooler \
  --projection_in_dim 768 \
  --projection_out_dim 128 \
  --deepquant_process \
  --deepquant_alpha 0.7 \
  --deepquant_interaction_mode disjoint_quantscorer \
  --colbert_normalize q \
  --colbert_rep_normalize \
  --colbert_temp 0.02 \
  --num_temp 0.02 \
  --quant_pairwiseprob_temp 1 \
  --quant_combinescore_temp 1 \
  --quant_relationpred_temp 1 \
  --deepquant_aggr_mode token \
  --deepquant_quantscoring_collate_mode max \
  --deepquant_quantscoring_combine_scores_mode attention_query \
  --deepquant_quantscoring_ver asym_dot \
  --asymdot_loss v2 \
  --process_whitespace \
  --num_as_token \
  --numgpt_embed \
  --new_mode \
  --embed_exp_frac 0.25 \
  --num_pred_loss_alpha 1 \
  --add_num_pred_loss \
  --DEBUG_use_rawnum \
  --op_pred_mode pred_noloss \
  --score_combine linear \
  --DEBUG_pairprob pred \
  --deepquant_alpha_mode cls \
  --DEBUG_score_relpred_num disjoint_loss_bert \
  --REMOVE_pairwise_prog_norm softmax \
  --REMOVE_pairwise_prog_score dot \
  --num_get_unit_ctx \
  --inject_units \
  --add_relation_classifier_loss \
  --add_num_relation_pred_loss \
  --all_units $dataroot/all_units.pickle \
  --equal_score asym_dot | tee $path/train_logs.log