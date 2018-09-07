from setuptools import setup

setup(
    name='mlpython',
    version='0.1.3',
    packages=['', 'mlpython', 'mlpython.doc', 'mlpython.misc', 'mlpython.misc.third_party',
              'mlpython.misc.third_party.tsne', 'mlpython.scripts', 'mlpython.datasets', 'mlpython.learners',
              'mlpython.learners.sparse', 'mlpython.learners.third_party', 'mlpython.learners.third_party.gpu',
              'mlpython.learners.third_party.milk', 'mlpython.learners.third_party.libsvm',
              'mlpython.learners.third_party.orange', 'mlpython.learners.third_party.word2vec',
              'mlpython.learners.third_party.treelearn', 'mlpython.mathutils', 'mlpython.mlproblems',
              'mlpython.unit_tests', 'mlpython.unit_tests.test_datasets', 'mlpython.unit_tests.test_learners',
              'mlpython.unit_tests.test_mathutils', 'mlpython.unit_tests.test_mlproblems'],
    url='https://github.com/AmirHooshangi/mlpython',
    license='GPL',
    author='Hugo Larochelle',
    author_email='hugo.larochelle@usherbrooke.ca ',
    description='Small ML library for loading standard ML datasets and models'
)
