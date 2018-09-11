# alexioannides

Python source code for generating my personal website - hosted by GitHub pages at [alexioannides.github.io](http://alexioannides.github.io) - using the [Pelican](https://blog.getpelican.com/) framework for static websites, together with [Flex theme](https://github.com/alexandrevicenzi/flex).

The output of the build process is written to the `output` folder in the root directory, that is **not** version controlled using this repository. Instead, the `output` directory has its own repository at [alexioannides](https://github.com/AlexIoannides/alexioannides), that is necessary for hosting with GitHub pages.

## Managing Dependencies

We use [pipenv](https://docs.pipenv.org) for managing project dependencies and Python environments (i.e. virtual environments). All of the direct packages dependencies required to run the code (i.e. Pelican and Markdown), are described in the `Pipfile`. Their precise downstream dependencies are described in `Pipfile.lock`.

### Installing Pipenv

To get started with Pipenv, first of all download it - assuming that there is a global version of Python available on your system and on the PATH, then this can be achieved by running the following command,

```bash
pip3 install pipenv
```

Pipenv is also available to install from many non-Python package managers. For example, on OS X it can be installed using the [Homebrew](https://brew.sh) package manager, with the following terminal command,

```bash
brew install pipenv
```

For more information, including advanced configuration options, see the [official pipenv documentation](https://docs.pipenv.org).

### Installing this Projects' Dependencies

Make sure that you're in the project's root directory (the same one in which `Pipfile` resides), and then run,

```bash
pipenv install
```

## Building the Website

To build the website we need to call Pelican,

```bash
pipenv run pelican
```

## Testing the Site Locally

We recommend setting `RELATIVE_URLS = True` when testing (do not forget to revert this before deploying) and then executing the following,

```bash
cd output
pipenv run python -m pelican.server
```

## Deploying to GutHub Pages

After testing locally, first of all ensure that `RELATIVE_URLS = False`, rebuilding the website if necessary. Then, make sure that you are still in the `output` directory and remember that this is version controlled by a [different repository](https://github.com/AlexIoannides/alexioannides), that now needs new changes to be committed and pushed to `master` as usual - e.g.,

```bash
git add -A
git commit -m "latest changes to alexioannides.github.io"
git push origin master
```

The [updated website](http://alexioannides.github.io) is usually available within a minute or two.
