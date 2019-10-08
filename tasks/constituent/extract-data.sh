cd ../../jiant/probing/data
python extract_ontonotes_all.py \
  --ontonotes /share/data/lang/users/freda/codebase/hackathon_2019/tasks/constituent/data/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0 \
  --tasks const coref ner srl \
  --splits train development test conll-2012-test \
  -o /share/data/lang/users/freda/codebase/hackathon_2019/tasks/constituent/data/edges/ontonotes