# -*- coding: utf-8 -*-

__version__ = 0.1
__author__ = 'Ying-Qiu Zheng'
__copyright__ = 'Copyright 2018, Ying-Qiu Zheng'
__credits__ = ['Ying-Qiu Zheng']
__license__ = 'GPLv2'
__email__ = 'contact@yingqiuzheng.me'
__status__ = 'Prototype'
__url__ = 'http://github.com/yingqiuz/epidemia'
__packagename__ = 'epidemia'
__description__ = (
    'Epidemia is a Python Toolbox for '
    'Agent-Based Epidemic Modelling on Networks.'
)
__longdesc__ = 'README.md'
__longdesctype__ = 'text/markdown'


REQUIRES = [
    'numpy',
    'scipy',
    'tqdm'
]

TESTS_REQUIRE = [
    'pytest',
    'pytest-cov'
]

EXTRAS_REQUIRE = {
    'dev': [
        'coverage',
        'flake8',
    ],
    'plotting': [
        'pandas',
        'seaborn'
    ],
    'doc': [
        'sphinx>=1.2',
        'sphinx_rtd_theme'
    ],
    'tests': TESTS_REQUIRE,
}

EXTRAS_REQUIRE['all'] = list(
    set([v for deps in EXTRAS_REQUIRE.values() for v in deps])
)

CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
]
