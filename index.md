# Newfrac computational platform website

This website a Jupyter-book

## Usage

### Building the book

If you'd like to develop on and build the my-book book, you should:

- Clone this repository and run
- Run `pip install -r requirements.txt` (it is recommended you do this within a virtual environment)
- (Recommended) Remove the existing `book/_build/` directory
- Run `jupyter-book/`
 build boo
A fully-rendered HTML version of the book will be built in `book/_build/html/`.

### Hosting the book

The html version of the book is hosted on the gitlab Pages branch of this repo. A Gitlab CI workflow has been created that automatically builds and pushes the book to this branch on a push or pull request to main.

This will automatically push your build to the Gitlab CI. More information on this hosting process can be found [here](https://jupyterbook.org/publish/gh-pages.html#manually-host-your-book-with-github-pages).

### How to contribute

All the material of the webpage is in the [book](https://gitlab.com/newfrac/newfrac.gitlab.io/-/tree/master/book) directory.

- To modify a page: just edit the .md file regenerate the book (locally by running `jupyter-book build book/` or authomatically with the `CI` in gitlab )
- To add a page: add the file and/or directory at the appropriate place inside the [book](https://gitlab.com/newfrac/newfrac.gitlab.io/-/tree/master/book) directory and update the [_toc.yml](https://gitlab.com/newfrac/newfrac.gitlab.io/-/blob/master/book/_toc.yml) to add the file/directory in the appropriate place in the table-of-contents.

## Credits

This project is created using the open source [Jupyter Book project](https://jupyterbook.org/) and the book of [dolfinx-tutorial](https://github.com/jorgensd/dolfinx-tutorial/blob/dokken/jupyterbook/Dockerfile) as a template.
