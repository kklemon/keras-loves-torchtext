Keras ❤️ torchtext
=================

> Keras is love  
Keras is life  
Keras loves torchtext

[torchtext](https://github.com/pytorch/text) is a great library, putting a layer of abstraction over the usually very heavy data component in NLP projects, making the work with complex datasets a pace.
Sadly, as torchtext is based and built on PyTorch, using it with Keras is not directly possible.

_Keras ❤️ torchtext_ is to the rescue by providing lightweight wrappers for some Torchtext classes, making them easily work with Keras.

Installation
------------
```bash
pip install keras-loves-torchtext
```

Examples
--------
Wrap a `torchtext.data.Iterator` with `WrapIterator` and use it to train a Keras model:
```python
from torchtext.data import Dataset, Field, Iterator
from kltt import WrapIterator

...

fields  = [('text', Field()),
           ('label', Field(sequential=False))]
dataset = Dataset(examples, fields)
iterator = Iterator(dataset, batch_size=32)

# Keras ❤️ torchtext comes to play
data_gen = WrapIterator(iterator, x_fields=['text'], y_fields=['label'])

model.fit_generator(iter(data_gen), steps_per_epoch=len(data_gen))
```


Easily wrap multiple iterators at once:
```python
from torchtext.data import Dataset, Field, Iterator
from kltt import WrapIterator

...

fields  = [('text', Field()),
           ('label', Field(sequential=False))]
dataset = Dataset(examples, fields)
splits = dataset.split()

iterators = Iterator.splits(splits, batch_size=32)
train, valid, test = WrapIterator.wraps(iterators, x_fields=['text'], y_fields=['label'])
model.fit_generator(iter(train), steps_per_epoch=len(train),
                    validation_data=iter(valid), validation_steps=len(valid))
loss, acc = model.evaluate_generator(iter(test), steps=len(test))
```

Further and full working examples can be found in the `examples` folder. 

Documentation
-------------
Todo

See `examples` and inline documentation for now.
