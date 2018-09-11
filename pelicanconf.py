#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

# Uncomment following line if you want document-relative URLs when developing
# RELATIVE_URLS = True

PATH = 'content'
STATIC_PATHS = ['images', 'extra']

OUTPUT_PATH = 'output/'
DELETE_OUTPUT_DIRECTORY = True
OUTPUT_RETENTION = ['.git', 'README.md', 'CNAME']

THEME = 'themes/Flex'
EXTRA_PATH_METADATA = {
    'extra/custom.css': {'path': 'static/custom.css'},
}
CUSTOM_CSS = 'static/custom.css'

AUTHOR = 'Dr Alex Ioannides'
SITEURL = 'https://alexioannides.github.io'
SITENAME = AUTHOR
SITETITLE = AUTHOR
SITESUBTITLE = 'Financial engineer - (data) scientist - habitual coder'
SITEDESCRIPION = 'Alex Ioannides on data science: data mining, statistics, machine learning, AI, functional programming, R, Python, Scala, Spark, Elasticsearch, AWS, DevOps...'
SITELOGO = '//avatars1.githubusercontent.com/u/5968486?s=460&v=4'
FAVICON = '/images/favicon.ico'
PYGMENTS_STYLE = 'monokai'
ROBOTS = 'index, follow'

TIMEZONE = 'Europe/London'
DEFAULT_LANG = 'en'

DEFAULT_PAGINATION = 15
TYPOGRIFY = True

USE_FOLDER_AS_CATEGORY = True
DEFAULT_CATEGORY = 'misc'

MAIN_MENU = True
MENUITEMS = (('Categories', '/categories.html'),
             ('Tags', '/tags.html'),
             ('Archives', '/archives.html'))

SOCIAL = (('github', 'https://github.com/alexioannides'),
          ('linkedin', 'https://www.linkedin.com/in/alexioannides/'),
          ('twitter', 'https://twitter.com/ioannides_alex'),
          ('soundcloud', 'https://soundcloud.com/user-616657739'))

# Feed generation is usually not desired when developing
FEED_DOMAIN = SITEURL
FEED_ALL_ATOM = 'feeds/all-atom.xml'
CATEGORY_FEED_ATOM = 'feeds/%s-atom.xml'
FEED_ALL_RSS = 'feeds/all-rss.xml'
CATEGORY_FEED_RSS = 'feeds/%s-rss.xml'

# Blogroll
# LINKS = (('Pelican', 'http://getpelican.com/'),
#          ('Python.org', 'http://python.org/'),
#          ('Jinja2', 'http://jinja.pocoo.org/'))

# Flex theme integrations
DISQUS_SITENAME = 'alexioannides'
GOOGLE_ANALYTICS= 'UA-125604661-1'