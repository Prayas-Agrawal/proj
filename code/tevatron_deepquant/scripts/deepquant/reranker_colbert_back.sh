root=/mnt/nas/user/tevatron/outputs/deepquant
path=$root/clinical2/relraw_oppred_asym_dot_pairprobPREDLOSSv1
dataroot=/mnt/nas/user/QuantityAwareRankers/train_data/clinical/final
data=$path/quant_clinical_alpha_rerank_out_file.bm25.monobert_norm_bm25_5000.tsv
async_log=$path/ids_alpha_v2.csv
# data=$path/FIN.jsonl
eval "$(conda shell.bash hook)"
conda activate /mnt/nas/user/condaEnvV3 &&

mkdir -p $path &&
SOURCE=$0
TS=$(date +"%m-%d_%H-%M-%S")
DEST_PATH="$path/$(basename $SOURCE .sh)_$TS.sh"
cp $SOURCE $DEST_PATH
echo "Script copied successfully to $DEST_PATH"

CUDA_VISIBLE_DEVICES=2 python3 -m tevatron.reranker.reranker_inference \
  --output_dir=temp \
  --model_name_or_path $path \
  --tokenizer_name bert-base-uncased \
  --save_strategy epoch \
  --dataset_name Tevatron/msmarco-passage \
  --encoded_save_path $data \
  --encode_in_path /mnt/nas/user/QuantityAwareRankers/train_data/clinical/tevatron_files/rerank_normalized/bm25_5000.jsonl \
  --dataset_format json \
  --bf16 \
  --encoded_format overload \
  --per_device_eval_batch_size 156 \
  --q_max_len 32 \
  --q_len 32 \
  --p_max_len 128 \
  --p_len 128 \
  --overwrite_output_dir \
  --data_cache_dir /mnt/nas/user/.huggingfaceCachecolb_token_clinical/ \
  --cache_dir /mnt/nas/user/.huggingfaceCachecolb_token_clinical/ \
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
  --deepquant_alpha_mode dot_cls \
  --DEBUG_score_relpred_num disjoint_loss_bert \
  --REMOVE_pairwise_prog_norm softmax \
  --REMOVE_pairwise_prog_score dot \
  --num_get_unit_ctx \
  --inject_units \
  --add_num_relation_pred_loss \
  --add_relation_classifier_loss \
  --async_log $async_log \
  --equal_score asym_dot | tee $path/rerank_logs2.log