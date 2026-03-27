cd ..
python -m scripts.generate_abscon --folder_path results/Meta-Llama-3.1-70B-Instruct --dataset_name clevr --num_candidates_start 1 --num_candidates_end 20
# python -m scripts.generate_abscon --folder_path results/Meta-Llama-3.1-8B-Instruct --dataset_name clevr --num_candidates_start 13 --num_candidates_end 20
# python -m scripts.generate_abscon --folder_path results/gpt-4o-mini --dataset_name clevr --num_generations $num_generations
