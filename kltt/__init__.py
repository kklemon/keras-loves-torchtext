from torchtext.data import Field


def _depth(obj):
    """Helper function to determine the depth of a nested list structure."""
    return isinstance(obj, (list, tuple)) and max(map(_depth, obj)) + 1


class WrapIterator(object):
    """
    Wraps a `torchtext.data.Iterator` to be used as data generator with Keras.

    Arguments:
        iterator: `torchtext.data.Iterator` instance to be wrapped.
        x_fields: Can be used to specify which field names correspond to input data. If None, Field names with
        is_target attribute set to False will be considered as input fields.
        y_fields: Can be used to specify which field names correspond to target data. If None, Field names with
        is_target attribute set to True will be considered as target fields.
        permute: Either None or a dictionary where each key is a field name that points to a list of dimension indices
        by which the corresponding output tensors should be permuted.

    Example:
        >>> dataset = Dataset(examples, [('text', text), ('label', label)])
        >>> iterator = Iterator(dataset, batch_size=32)
        >>> data_gen = WrapIterator(iterator)
        >>> model.fit_generator(iter(data_gen), steps_per_epoch=len(data_gen))
    """
    def __init__(self, iterator, x_fields=None, y_fields=None, permute=None):
        self.iterator = iterator
        self.permute = permute
        self.x_fields = []
        self.y_fields = []

        self.x_fields = self._process_fields_argument(x_fields, False)
        self.y_fields = self._process_fields_argument(y_fields, True)

    def _process_fields_argument(self, field_names, is_target):
        result = []
        if field_names is None:
            for name, field in self.iterator.dataset.fields.items():
                if issubclass(field.__class__, Field):
                    if not hasattr(field, 'is_target'):
                        raise Exception('Field instance does not have a is_target attribute and input and output '
                                        'fields also haven\'t been provided specifically.'
                                        ' Consider to upgrade torchtext or provide all input and output fields '
                                        'with the x_fields and y_fields arguments.')
                    if field.is_target == is_target:
                        result.append(name)
        else:
            all_field_names = list(self.iterator.dataset.fields.keys())
            for name in field_names:
                if name not in all_field_names:
                    raise ValueError('Provided input field \'{}\' is not in dataset\'s field list'.format(name))
            result.extend(field_names)

        if not result:
            raise Exception('No {} fields have been provided. Either provide fields with is_target attribute set to '
                            '{} or pass a list of field names as the {} argument'
                            .format('target' if is_target else 'input',
                                    is_target,
                                    'y_fields' if is_target else 'x_fields'))
        return result

    @classmethod
    def wraps(cls, iterators, x_fields=None, y_fields=None, **kwargs):
        """
        Wrap multiple iterators.

        Arguments:
            iterators: List of iterators to wrap.
            x_fields: Can be used to specify which field names correspond to input data. If None, Field names with
            is_target attribute set to False will be considered as input fields.
            y_fields: Can be used to specify which field names correspond to target data. If None, Field names with
            is_target attribute set to True will be considered as target fields.
            **kwargs: Arguments that will be passed to the constructor of `WrapIterator` instances.

        Example:
            >>> splits = Dataset.splits()
            >>> iterators = Iterator.splits(splits, batch_size=32)
            >>> train, test = WrapIterator(iterators)
            >>> model.fit_generator(iter(train), steps_per_epoch=len(train))
            >>> model.evaluate_generator(iter(test), steps=len(test))
        """
        def process_fields(fields, name):
            if fields is not None:
                depth = _depth(fields)
                if depth not in [1, 2]:
                    raise ValueError('\'{}\' must be either a list of field names or a list of'
                                     ' field name lists, one for each iterator'.format(name))
                fields = fields if depth == 2 else [fields] * len(iterators)
            return fields

        x_fields = process_fields(x_fields, 'x_fields')
        y_fields = process_fields(y_fields, 'y_fields')

        wrappers = []
        for i, it in enumerate(iterators):
            x_fields_arg = x_fields[i] if x_fields else None
            y_fields_arg = y_fields[i] if y_fields else None
            wrappers.append(cls(it, x_fields_arg, y_fields_arg, **kwargs))
        return wrappers

    def _process(self, tensor, field_name):
        if self.permute and field_name in self.permute:
            tensor = tensor.permute(*self.permute[field_name])
        return tensor.cpu().numpy()

    def __iter__(self):
        for batch in iter(self.iterator):
            batch_x = [self._process(getattr(batch, field), field) for field in self.x_fields]
            batch_y = [self._process(getattr(batch, field), field) for field in self.y_fields]
            yield batch_x, batch_y

    def __len__(self):
        return len(self.iterator)
