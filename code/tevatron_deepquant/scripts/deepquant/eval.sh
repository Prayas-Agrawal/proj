root=/mnt/nas/user/tevatron/outputs/deepquant
# path=$root/test_finalv2_yesnumrelpred_singlestep_project_7thepoch

# path=$root/bert_test/relraw_oppred_asym_dot_pairprobPREDLOSSv1softmax-dot-alpha-dot_cls-NOTEMP-fixed
path=$root/test_final_clsalpha_v2
# path=$root/colbert_deepquant_fin2/numgpt_relevance_qcolb_opinversion/lossv2_alpha0.7_max_collate_attention_query_FFL_aggr_token
# path=$root/colbert_deepquant_fin2/qcolb_test/lossv2_alpha0.7_max_collate_attention_query_FFL_aggr_token

# results=/mnt/nas/user/tevatron/src/tevatron/modeling/clinical_reranked_1000_norm.tsv

# results=/mnt/nas/user/rank_llm/rerank_results/SPLADE_P_P_ENSEMBLE_DISTIL/test/TEV_rank_zephyr_7b_v1_full_4096_1000_rank_GPT_deepquantCLINICAL_p1_2025-02-13T03:02:56.985510_window_20.tsv
results=$path/quant_finance_alpha_v4.tsv

out=$path/eval_alpha
mode=finance

eval "$(conda shell.bash hook)" &&
conda activate /mnt/nas/user/condaEnvV3 &&
cd /mnt/nas/user/QuantityAwareRankers &&
python -m evaluate.evaluate_custom --ranker-result-path $results --output-folder $out --gt-folder ./data/$mode/proc --queries-path ./data/$mode/queries.tsv