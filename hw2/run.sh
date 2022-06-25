python multiple_choice.py --model_name_or_path ./multiChoice/macbertLarge/ --do_predict --output_dir ./multiChoice/macbertLarge --context_file $1 --test_file $2 --max_seq_length 512 --pad_to_max_length True --cache_dir ./cache --output_file ./multiChoicePredictMacbert.json

python run_qa.py --model_name_or_path ./qa/macbertLarge/ --do_predict --test_file ./multiChoicePredictMacbert.json --context_file $1 --max_seq_length 512 --output_dir ./qa/macbertLarge/ --output_file ./qa_prediction_macbert.json --overwrite_output_dir --overwrite_output

mv ./submission.csv $3