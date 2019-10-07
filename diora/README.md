## DIORA Span Representation

#### Dependencies
```
conda create -n diora python=3.6
source activate diora

conda install pytorch==1.1.0 torchvision -c pytorch
pip install allennlp
pip install tqdm

(clone this repo)
cd pytorch
export PYTHONPATH=$(pwd):$PYTHONPATH
bash config.sh 
```

Then you should be able to run the following script anywhere:
```Python
from diora_span import DioraRepresentation

sent = 'hello world !'.split()
tool = DioraRepresentation(PATH_TO_YOUR_CORPUS)
representation = tool.span_representation(sent, 0, 3)
```

Note that the implementation of DIORA requires the whole corpus before calculating span representation, i.e., no UNK is allowed. 
For example, in order to run the scrpit above, your corpus should at least have words "hello", "world" and "!"
Your corpus could just be some (tokenized) sentences, e.g., one line each. 

Also, the trained DIORA model requires lowercase letters -- while you could keep the corpus case-sensitive, I generally suggest making them lowercase to ensure that the model is looking at in-domain sentences.
