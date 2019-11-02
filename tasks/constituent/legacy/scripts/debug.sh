python -m tasks.constituent.main --model-name debug-01 \
    --model-type bert --model-size base --encoding-method coherent --epoch-run 2 --epochs 5 \
    --data-path tasks/constituent/data/edges/ontonotes/const/debug --eval-step 2 --log-step 1 --use-proj 

python -m tasks.constituent.main --model-name debug-01 \
    --model-type bert --model-size base --encoding-method coherent --epoch-run 5 --epochs 5 \
    --data-path tasks/constituent/data/edges/ontonotes/const/debug --eval-step 2 --log-step 1 --use-proj
