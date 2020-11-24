Topic Modelling is the task of using unsupervised learning to extract the main topics (represented as a set of words) 
that occur in a collection of documents.


# Test the project


```
pip install -r requirements.txt
```

In a Python3 environment:
```python
from topics_finder.models import LsaModel
lsa_obj = LsaModel()
lsa_obj.train_predict()
```

