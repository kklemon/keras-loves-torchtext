def _depth(obj):
    """Helper function to determine the depth of a nested list structure."""
    return isinstance(obj, (list, tuple)) and max(map(_depth, obj)) + 1


class WrapIterator(object):
    """
    Wraps a `torchtext.data.Iterator` to be used as data generator with Keras.

    Arguments:
        iterator: `torchtext.data.Iterator` instance to be wrapped.
        x_fields: List of field names that correspond to input data.
        y_fields: List of field names that correspond to output data.
        permute: Either None or a dictionary where each key is a field name that points to a list of dimension indices
        by which the corresponding output tensors should be permuted.

    Example:
        >>> dataset = Dataset(examples, [('text', text), ('label'. label)])
        >>> iterator = Iterator(dataset, batch_size=32)
        >>> data_gen = WrapIterator(iterator, ['text'], ['label'])
        >>> model.fit_generator(iter(data_gen), steps_per_epoch=len(data_gen))
    """
    def __init__(self, iterator, x_fields, y_fields, permute=None):
        self.iterator = iterator
        self.x_fields = x_fields
        self.y_fields = y_fields
        self.permute = permute

    @classmethod
    def wraps(cls, iterators, x_fields, y_fields, **kwargs):
        """
        Wrap multiple iterators.

        Arguments:
            iterators: List of iterators to wrap.
            x_fields: Field names corresponding to the input data. Either a list of field names that will be applied
            to all iterators or a list with field name lists for each iterator.
            y_fields: Field names corresponding to the output data. Either a list of field names that will be applied
            to all iterators or a list with field name lists for each iterator.
            **kwargs: Arguments that will be passed to the constructor of `WrapIterator` instances.

        Example:
            >>> splits = Dataset.splits()
            >>> iterators = Iterator.splits(splits, batch_size=32)
            >>> train, test = WrapIterator(iterators, ['text'], ['label'])
            >>> model.fit_generator(iter(train), steps_per_epoch=len(train))
            >>> model.evaluate_generator(iter(test), steps=len(test))
        """
        for fields, name in zip([x_fields, y_fields], ['x_fields', 'y_fields']):
            if _depth(fields) not in [1, 2]:
                raise ValueError('\'{}\' must be either a list of field names or a list of'
                                 ' field name lists, one for each iterator'.format(name))

        x_fields = x_fields if _depth(x_fields == 2) else [x_fields] * len(iterators)
        y_fields = y_fields if _depth(y_fields == 2) else [y_fields] * len(iterators)

        wrappers = []
        for it, x_fields_it, y_fields_it in zip(iterators, x_fields, y_fields):
            wrappers.append(cls(it, x_fields_it, y_fields_it, **kwargs))
        return wrappers

    def _process(self, tensor, field_name):
        tensor = tensor.permute(*self.permute[field_name]) if field_name in self.permute else tensor
        return tensor.cpu().numpy()

    def __iter__(self):
        for batch in iter(self.iterator):
            batch_x = [self._process(getattr(batch, field), field) for field in self.x_fields]
            batch_y = [self._process(getattr(batch, field), field) for field in self.y_fields]
            yield batch_x, batch_y

    def __len__(self):
        return len(self.iterator)
