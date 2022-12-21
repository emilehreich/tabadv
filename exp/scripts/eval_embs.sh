python eval.py --dataset syn --noise 85 --cost_bound 10 --attack greedy_delta --force --utility_type success_rate --model_path  lgbm 
python eval.py --dataset syn --noise 85 --cost_bound 10 --attack greedy_delta --force --utility_type success_rate --tr 0.0 --model_path  lgbm  --embs ../models/100.0_syn_85_embonly_long.pt121 
python eval.py --dataset syn --noise 85 --cost_bound 10 --attack greedy_delta --force --utility_type success_rate --tr 0.1 --model_path  lgbm  --embs ../models/100.0_syn_85_embonly_long.pt121 
python eval.py --dataset syn --noise 85 --cost_bound 10 --attack greedy_delta --force --utility_type success_rate --tr 0.2 --model_path  lgbm  --embs ../models/100.0_syn_85_embonly_long.pt121 
python eval.py --dataset syn --noise 85 --cost_bound 10 --attack greedy_delta --force --utility_type success_rate --tr 0.4 --model_path  lgbm  --embs ../models/100.0_syn_85_embonly_long.pt121 
python eval.py --dataset syn --noise 85 --cost_bound 10 --attack greedy_delta --force --utility_type success_rate --tr 1.0 --model_path  lgbm  --embs ../models/100.0_syn_85_embonly_long.pt121 
python eval.py --dataset syn --noise 85 --cost_bound 10 --attack greedy_delta --force --utility_type success_rate --tr 3.0 --model_path  lgbm  --embs ../models/100.0_syn_85_embonly_long.pt121 
python eval.py --dataset syn --noise 85 --cost_bound 10 --attack greedy_delta --force --utility_type success_rate --tr 10.0 --model_path  lgbm  --embs ../models/100.0_syn_85_embonly_long.pt121 
