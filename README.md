# One-shot Learning with Memory-Augmented Neural Networks
Theano implementation of the paper *One-shot Learning with Memory-Augmented Neural Networks*, by A. Santoro et al.

## Getting started
To avoid any conflict with your existing Python setup, and to keep this project self-contained, it is suggested to work in a virtual environment with [`virtualenv`](http://docs.python-guide.org/en/latest/dev/virtualenvs/). To install `virtualenv`:
```bash
sudo pip install --upgrade virtualenv
```

Create a virtual environment called `venv`, activate it and install the requirements given by `requirements.txt`.
```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Omniglot
In order to run the experiment on the [Omniglot dataset](https://github.com/brendenlake/omniglot), you first need to download the dataset in the [`data/omniglot`](data/omniglot/) folder (see the [`README`](data/omniglot/README.md) for more details).

## Tests
This projects has a few basic tests. To run these tests, you can run the `py.test` on the project folder
```bash
venv/bin/py.test mann -vv
```

## Paper
Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, Timothy Lillicrap, *One-shot Learning with Memory-Augmented Neural Networks*, [[arXiv](http://arxiv.org/abs/1605.06065)]
