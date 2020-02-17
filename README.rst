=============
CS272 Project
=============


.. image:: https://img.shields.io/pypi/v/cs272_project.svg
        :target: https://pypi.python.org/pypi/cs272_project

.. image:: https://img.shields.io/travis/taoyilee/cs272_project.svg
        :target: https://travis-ci.com/taoyilee/cs272_project

.. image:: https://readthedocs.org/projects/cs272-project/badge/?version=latest
        :target: https://cs272-project.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/taoyilee/cs272_project/shield.svg
     :target: https://pyup.io/repos/github/taoyilee/cs272_project/
     :alt: Updates



NLP Final Project


* Free software: MIT license
* Documentation: https://cs272-project.readthedocs.io.

Quick Start
-------------
Replication Only
======================
Following the steps below to setup training environment

.. code-block:: bash

    mkdir work_directory
    cd work_directory
    # create virtual environment under work_directory, naming it to "venv"
    python -m venv venv
    source venv/bin/activate
    # install the package
    pip install cs272-project
    # write configuration file to a working directory
    cs272_project_cli write-config --outfile /home/tylee/PycharmProjects/nlp_workspace

Development
======================
First of all, please fork the project if you are interested in extending its functionalities.
After that, you may clone the repository with:

.. code-block:: bash

    git clone git@github.com:<user_name>/nlp_project.git

Pull requests welcome!

Dependencies
======================
1. `PyTorch <https://pytorch.org/>`_ == 1.4.0
2. `YouTokenToMe <https://github.com/VKCOM/YouTokenToMe>`_ (yttm) == 1.0.6
3. `Huggingface Transformers <https://github.com/huggingface/transformers>`_ == 2.4.1
4. Click >= 0.7

Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
